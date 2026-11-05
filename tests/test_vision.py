"""Regression tests with a loopback HTTP server; no real key or remote API."""
import base64
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
import brotli
import numpy as np
from PIL import Image

PACKAGE = ROOT
spec = importlib.util.spec_from_file_location('gpt_vision', PACKAGE / '__init__.py',
                                             submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules['gpt_vision'] = package
spec.loader.exec_module(package)
from gpt_vision import nodes
from gpt_vision import vision_client as client


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.seen.append((self.path, dict(self.headers), json.loads(body)))
        status, headers, response = self.server.reply
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_):
        pass


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'
        cls.env = patch.dict(os.environ, {'NO_PROXY': '127.0.0.1,localhost',
                                         'COMFY_GPT_VISION_API_KEY': ''})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.env.stop()

    def setUp(self):
        self.server.seen = []
        self.reply({'choices': [{'message': {'content': 'geometry analysis'},
                                 'finish_reason': 'stop'}]})

    def reply(self, obj=None, encoding='', raw=None, status=200):
        if raw is None:
            raw = json.dumps(obj, ensure_ascii=False).encode()
        self.server.reply = (status, {'Content-Type': 'application/json',
                                     'Content-Encoding': encoding}, raw)

    def run_node(self, **overrides):
        args = dict(prompt='Analyze product geometry', model='provider-model-id',
                    base_url=self.url, api_mode='chat_completions', api_key='unit-test-key',
                    token_parameter='max_completion_tokens', max_output_tokens=4096,
                    image_detail='auto', max_image_side=1536, timeout_seconds=10,
                    refresh_seed=0)
        args.update(overrides)
        return nodes.GPTVisionAnalyze().analyze(**args)

    def test_six_images_all_sent_in_order_and_long_text_preserved(self):
        text = '几何关系分析 ' * 400
        self.reply({'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]})
        batch = np.stack([np.full((8, 12, 3), i / 10, np.float32) for i in range(6)])
        result, debug = self.run_node(images=batch)
        self.assertEqual(result, text.strip())
        endpoint, headers, payload = self.server.seen[0]
        self.assertEqual(endpoint, '/v1/chat/completions')
        lower_headers = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(lower_headers['accept-encoding'], 'identity')
        self.assertFalse(payload['stream'])
        self.assertNotIn('temperature', payload)
        self.assertNotIn('frequency_penalty', payload)
        content = payload['messages'][0]['content']
        self.assertEqual(len(content), 7)
        for i, part in enumerate(content[1:]):
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                self.assertAlmostEqual(image.getpixel((0, 0))[0], round(i / 10 * 255), delta=1)
        self.assertEqual(json.loads(debug)['image_count'], 6)
        self.assertNotIn('unit-test-key', debug)

    def test_two_batches_and_no_upscale(self):
        a = np.zeros((2, 30, 60, 3), np.float32)
        b = np.zeros((1, 15, 30, 3), np.float32)
        _, debug = self.run_node(images=a, images_2=b, max_image_side=40)
        self.assertEqual(json.loads(debug)['sent_image_sizes'], [[40, 20], [40, 20], [30, 15]])

    def test_brotli_response_despite_identity(self):
        data = json.dumps({'choices': [{'message': {'content': 'Brotli 正常'}}]}).encode()
        self.reply(raw=brotli.compress(data), encoding='br')
        self.assertEqual(self.run_node()[0], 'Brotli 正常')

    def test_gzip_and_both_deflate_variants(self):
        data = json.dumps({'choices': [{'message': {'content': 'decoded'}}]}).encode()
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        raw_deflate = compressor.compress(data) + compressor.flush()
        for encoding, raw in [('gzip', gzip.compress(data)),
                              ('deflate', zlib.compress(data)), ('deflate', raw_deflate)]:
            with self.subTest(encoding=encoding, length=len(raw)):
                self.reply(raw=raw, encoding=encoding)
                self.assertEqual(self.run_node()[0], 'decoded')

    def test_bad_br_header_with_valid_uncompressed_json(self):
        self.reply({'choices': [{'message': {'content': 'valid plaintext JSON'}}]}, encoding='br')
        text, debug = self.run_node()
        self.assertEqual(text, 'valid plaintext JSON')
        self.assertTrue(json.loads(debug)['warnings'])

    def test_invalid_brotli_stops_node(self):
        self.reply(raw=b'not brotli or JSON', encoding='br')
        with self.assertRaisesRegex(RuntimeError, '压缩解码失败'):
            self.run_node()

    def test_missing_brotli_gives_installation_hint(self):
        real_import = client.importlib.import_module
        def missing(name, *args, **kwargs):
            if name in ('brotli', 'brotlicffi'):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)
        data = json.dumps({'choices': [{'message': {'content': 'ok'}}]}).encode()
        self.reply(raw=brotli.compress(data), encoding='br')
        with patch.object(client.importlib, 'import_module', side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, 'requirements.txt'):
                self.run_node()

    def test_responses_protocol_skips_reasoning_output(self):
        self.reply({'status': 'completed', 'output': [
            {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'not the answer'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'volume hierarchy'}]}]})
        text, _ = self.run_node(api_mode='responses', images=np.zeros((1, 4, 4, 3)))
        self.assertEqual(text, 'volume hierarchy')
        endpoint, _, payload = self.server.seen[0]
        self.assertEqual(endpoint, '/v1/responses')
        self.assertEqual(payload['max_output_tokens'], 4096)
        self.assertEqual(payload['input'][0]['content'][1]['type'], 'input_image')

    def test_full_endpoint_not_appended_twice(self):
        self.run_node(base_url=self.url + '/chat/completions')
        self.assertEqual(self.server.seen[0][0], '/v1/chat/completions')

    def test_endpoint_mode_mismatch_fails_before_request(self):
        with self.assertRaisesRegex(RuntimeError, '不一致'):
            self.run_node(base_url=self.url + '/responses')
        self.assertEqual(self.server.seen, [])

    def test_token_parameter_compatibility(self):
        self.run_node(token_parameter='max_tokens')
        payload = self.server.seen[-1][2]
        self.assertIn('max_tokens', payload)
        self.assertNotIn('max_completion_tokens', payload)
        self.run_node(token_parameter='omit')
        payload = self.server.seen[-1][2]
        self.assertNotIn('max_tokens', payload)
        self.assertNotIn('max_completion_tokens', payload)

    def test_401_redacts_key_and_stops(self):
        self.reply({'error': {'message': 'bad key unit-test-key sk-fake-token-for-tests'}}, status=401)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as caught:
            self.run_node()
        self.assertNotIn('unit-test-key', str(caught.exception))
        self.assertNotIn('sk-fake-token-for-tests', str(caught.exception))

    def test_non_json_html_stops(self):
        self.reply(raw=b'<html>proxy challenge page</html>')
        with self.assertRaisesRegex(RuntimeError, '有效 JSON'):
            self.run_node()

    def test_empty_and_truncated_output_stop(self):
        for choice in [{'message': {'content': ''}},
                       {'message': {'content': 'partial'}, 'finish_reason': 'length'}]:
            with self.subTest(choice=choice):
                self.reply({'choices': [choice]})
                with self.assertRaises(RuntimeError):
                    self.run_node()

    def test_timeout_does_not_retry(self):
        with patch.object(client.urllib.request.OpenerDirector, 'open', side_effect=socket.timeout) as opened:
            with self.assertRaisesRegex(RuntimeError, '超时'):
                self.run_node()
            self.assertEqual(opened.call_count, 1)

    def test_redirect_not_followed(self):
        self.server.reply = (302, {'Location': self.url + '/other'}, b'')
        with self.assertRaisesRegex(RuntimeError, 'HTTP 302'):
            self.run_node()
        self.assertEqual(len(self.server.seen), 1)

    def test_missing_key_and_oversized_batch_fail_before_network(self):
        with self.assertRaisesRegex(RuntimeError, 'api_key'):
            self.run_node(api_key='')
        with self.assertRaisesRegex(RuntimeError, '32'):
            self.run_node(images=np.zeros((33, 2, 2, 3)))
        self.assertEqual(self.server.seen, [])

    def test_env_key_is_specific_to_this_node(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'unrelated-provider-key',
                                    'COMFY_GPT_VISION_API_KEY': 'unit-test-key'}):
            self.run_node(api_key='')
        headers = {k.lower(): v for k, v in self.server.seen[0][1].items()}
        self.assertEqual(headers['authorization'], 'Bearer unit-test-key')


if __name__ == '__main__':
    unittest.main(verbosity=2)
