import base64
import io
import json
import os
import urllib.parse

import numpy as np
from PIL import Image

from .vision_client import VisionError, build_payload, endpoint_url, extract_text, post_json, redact


def encode_images(batches, max_image_side):
    urls, sizes = [], []
    for batch in batches:
        if batch is None:
            continue
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        array = np.asarray(batch)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
            raise VisionError("图片必须是 ComfyUI 的 BHWC IMAGE 张量。")
        if len(urls) + len(array) > 32:
            raise VisionError("单次最多支持 32 张图片，请拆分参考图；不会静默丢弃图片。")
        for frame in array:
            pixels = np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            image = Image.fromarray(pixels)
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image).convert("RGB")
            else:
                image = image.convert("RGB")
            if max_image_side > 0:
                image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
            sizes.append(list(image.size))
    return urls, sizes


class GPTVisionAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "Analyze the visible geometry and spatial relationships in the reference images. Do not infer hidden internal construction."}),
            "model": ("STRING", {"default": "gpt-6-astra"}),
            "base_url": ("STRING", {"default": ""}),
            "api_mode": (["chat_completions", "responses"],),
            "api_key": ("STRING", {"default": "", "tooltip": "可留空并在服务器设置 COMFY_GPT_VISION_API_KEY。填写在控件中的密钥可能随工作流导出。"}),
            "token_parameter": (["max_completion_tokens", "max_tokens", "omit"],),
            "max_output_tokens": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 256}),
            "image_detail": (["auto", "high", "low"],),
            "max_image_side": ("INT", {"default": 1536, "min": 0, "max": 8192, "step": 64, "tooltip": "最长边上限；0 保留原尺寸，不放大小图。"}),
            "timeout_seconds": ("INT", {"default": 180, "min": 10, "max": 600}),
            "refresh_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                     "control_after_generate": True, "tooltip": "只用于刷新 ComfyUI 缓存，不发送给模型。"}),
        }, "optional": {"images": ("IMAGE",), "images_2": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "diagnostics")
    FUNCTION = "analyze"
    CATEGORY = "GPT Vision"
    DESCRIPTION = "多图视觉分析。非流式，明确处理压缩响应；失败时抛错停止下游。"

    def analyze(self, prompt, model, base_url, api_mode, api_key, token_parameter,
                max_output_tokens, image_detail, max_image_side, timeout_seconds,
                refresh_seed, images=None, images_2=None):
        # Never fall back to an unrelated provider's OPENAI_API_KEY implicitly.
        key = (api_key or os.environ.get("COMFY_GPT_VISION_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("请填写本接口商的 api_key，或在 ComfyUI 服务器设置 COMFY_GPT_VISION_API_KEY。")
        try:
            url = endpoint_url(base_url, api_mode)
            image_urls, sizes = encode_images((images, images_2), int(max_image_side))
            payload = build_payload(model, prompt, image_urls, api_mode, token_parameter,
                                    max_output_tokens, image_detail)
            value, warnings = post_json(url, payload, key, int(timeout_seconds))
            full_text = extract_text(value, api_mode)
            full_text = full_text.replace(key, "[REDACTED]")
            debug = {"api_mode": api_mode, "endpoint": urllib.parse.urlsplit(url).netloc,
                     "model": model, "image_count": len(image_urls),
                     "sent_image_sizes": sizes, "stream": False,
                     "accept_encoding": "identity", "warnings": warnings}
            return (full_text, json.dumps(debug, ensure_ascii=False, indent=2))
        except Exception as exc:
            # Upstream errors can echo request fields; do not log raw payloads.
            raise RuntimeError(redact(str(exc), key)) from None
