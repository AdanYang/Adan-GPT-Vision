"""Small, non-streaming OpenAI-compatible client. No SDK or aiohttp required."""

import gzip
import importlib
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib


MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class VisionError(RuntimeError):
    pass


def redact(message, secret=""):
    text = str(message)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    text = re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", text)
    return text[:1200]


def endpoint_url(base_url, api_mode):
    url = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise VisionError("base_url 必须是完整的 http:// 或 https:// 地址。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VisionError("base_url 不接受用户名、密码、查询参数或片段；密钥请填在 api_key。")
    if api_mode not in ("chat_completions", "responses"):
        raise VisionError("不支持的 api_mode。")
    suffix = "/chat/completions" if api_mode == "chat_completions" else "/responses"
    for ending in ("/chat/completions", "/responses"):
        if parsed.path.endswith(ending):
            if ending != suffix:
                raise VisionError("完整接口地址与 api_mode 不一致，请修改其中一项。")
            return url
    return url + suffix


def build_payload(model, prompt, image_urls, api_mode, token_parameter,
                  max_output_tokens, image_detail):
    if not model.strip() or not prompt.strip():
        raise VisionError("model 和 prompt 不能为空。")
    if token_parameter not in ("max_completion_tokens", "max_tokens", "omit"):
        raise VisionError("不支持的 token_parameter。")
    if image_detail not in ("auto", "high", "low"):
        raise VisionError("不支持的 image_detail。")
    payload = {"model": model.strip(), "stream": False}
    if api_mode == "chat_completions":
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {
            "url": url, "detail": image_detail}} for url in image_urls)
        payload["messages"] = [{"role": "user", "content": content}]
        if token_parameter != "omit":
            payload[token_parameter] = int(max_output_tokens)
    else:
        content = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": url,
                        "detail": image_detail} for url in image_urls)
        payload["input"] = [{"role": "user", "content": content}]
        if token_parameter != "omit":
            payload["max_output_tokens"] = int(max_output_tokens)
    return payload


def _brotli_decompress(data):
    for name in ("brotli", "brotlicffi"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        return module.decompress(data)
    raise VisionError(
        "服务端仍返回 br 压缩。请在运行 ComfyUI 的同一 Python 环境安装本节点的 "
        "requirements.txt 并重启，或让接口商关闭该接口的 Brotli 压缩。")


def decode_json_response(raw, encoding):
    """urllib leaves content-encoding untouched, so decode it explicitly."""
    warnings = []
    try:
        decoded = raw
        for coding in reversed([c.strip().lower() for c in encoding.split(",") if c.strip()]):
            if coding == "identity":
                continue
            if coding == "gzip":
                decoded = gzip.decompress(decoded)
            elif coding == "deflate":
                try:
                    decoded = zlib.decompress(decoded)
                except zlib.error:
                    decoded = zlib.decompress(decoded, -zlib.MAX_WBITS)
            elif coding == "br":
                decoded = _brotli_decompress(decoded)
            else:
                raise VisionError("接口返回不支持的压缩编码：" + coding)
            if len(decoded) > MAX_RESPONSE_BYTES:
                raise VisionError("解压后的响应过大，已停止解析。")
    except Exception as exc:
        # Some gateways mark an already-decoded JSON body as br. Only recover
        # when the complete original body really is valid JSON.
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (ValueError, UnicodeError):
            if isinstance(exc, VisionError):
                raise exc from None
            raise VisionError(
                "响应压缩解码失败（" + encoding + "）。请检查接口响应头、压缩数据及 Brotli 依赖。"
            ) from None
        warnings.append("接口压缩响应头与正文不一致；已按有效的未压缩 JSON 读取。")
        return value, warnings
    try:
        return json.loads(decoded.decode("utf-8-sig")), warnings
    except (ValueError, UnicodeError):
        raise VisionError(
            "接口未返回有效 JSON。可能返回了流式 SSE、HTML 登录/拦截页或不完整响应；"
            "请检查接口地址和代理。请求已设置 stream=false。") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_response(response):
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise VisionError("接口响应超过 16 MiB，已停止读取。")
    return decode_json_response(raw, response.headers.get("Content-Encoding", ""))


def post_json(url, payload, api_key, timeout_seconds):
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": "application/json", "Accept": "application/json",
                 "Accept-Encoding": "identity", "User-Agent": "ComfyUI-GPT-Vision/1.0"},
        method="POST")
    # No auto-retry: a timed-out request may already have been billed.
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            value, warnings = _read_response(response)
            return value, warnings
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            value, _ = _read_response(exc)
            error = value.get("error", value) if isinstance(value, dict) else value
            if isinstance(error, dict):
                detail = error.get("message", error.get("code", "接口拒绝请求"))
            else:
                detail = str(error)
        except VisionError as decode_error:
            detail = str(decode_error)
        finally:
            exc.close()
        hints = {401: "检查接口商的密钥。", 403: "检查模型或账户权限。",
                 404: "检查 base_url、api_mode 和模型 ID。",
                 429: "额度不足或限流，请查看接口商控制台。"}
        if 300 <= status < 400:
            detail = "接口要求跳转。请填写最终 API 地址；节点不会携带密钥跟随跳转。"
        raise VisionError(redact(f"HTTP {status}: {detail} {hints.get(status, '')}", api_key)) from None
    except (socket.timeout, TimeoutError):
        raise VisionError("接口请求超时。未自动重试；请先查看接口商任务或用量记录。") from None
    except urllib.error.URLError as exc:
        raise VisionError(redact("无法连接接口：" + str(exc.reason), api_key)) from None


def extract_text(value, api_mode):
    if not isinstance(value, dict):
        raise VisionError("接口返回的 JSON 顶层不是对象。")
    if value.get("error"):
        error = value["error"]
        raise VisionError(str(error.get("message", error)) if isinstance(error, dict) else str(error))
    if api_mode == "chat_completions":
        choices = value.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise VisionError("响应缺少 choices；请确认该接口支持 Chat Completions。")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise VisionError("输出达到 token 上限而被截断，请提高 max_output_tokens 后重试。")
        message = choice.get("message") or {}
        if message.get("refusal"):
            raise VisionError("模型拒绝处理此请求：" + str(message["refusal"]))
        text = message.get("content")
        if isinstance(text, list):
            text = "\n".join(p.get("text", "") for p in text
                             if isinstance(p, dict) and isinstance(p.get("text"), str))
    else:
        if value.get("status") in ("incomplete", "failed", "cancelled", "queued", "in_progress"):
            raise VisionError("Responses 请求尚未成功完成：" + str(value.get("status")) +
                              "；请检查 token 上限或接口是否只支持异步任务。")
        text = value.get("output_text")
        if not isinstance(text, str) or not text.strip():
            parts = []
            for item in value.get("output", []):
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise VisionError("模型拒绝处理此请求：" + str(part.get("refusal", "")))
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            text = "\n".join(parts)
    if not isinstance(text, str) or not text.strip():
        raise VisionError("接口没有返回可用的分析文字；不会把空结果传给生图节点。")
    return text.strip()
