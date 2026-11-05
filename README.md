# Adan-GPT-Vision

支持 OpenAI 兼容接口的 ComfyUI 多图视觉分析节点。节点类型为 `GPTVisionAnalyze`，菜单名称为 **GPT 视觉分析 · 非流式 / 多图**。

本项目是节点源码，不包含模型权重、API Key 或调用额度。必须自行选择支持图片输入的模型与兼容接口。

## 功能

- Chat Completions 和 Responses 两种协议。
- 一次请求发送完整图片 batch，最多 32 张，支持第二路 batch 输入。
- 非流式请求，设置 `Accept-Encoding: identity`。
- 明确处理 gzip、deflate、Brotli 响应；仅在正文为有效 JSON 时恢复错误压缩头。
- 请求错误、空结果、截断输出抛出异常，停止下游。
- 诊断输出显示发送图片数、尺寸和压缩头异常提示。
- 不自动重试，不保存对话历史，不自动读取其他服务的 OPENAI_API_KEY。

## 本地安装

仓库地址：<https://github.com/AdanYang/Adan-GPT-Vision>

将仓库克隆到 `ComfyUI/custom_nodes/Adan-GPT-Vision/`，使用运行 ComfyUI 的同一个 Python 安装依赖后重启：

```text
git clone https://github.com/AdanYang/Adan-GPT-Vision.git ComfyUI/custom_nodes/Adan-GPT-Vision
python -m pip install -r ComfyUI/custom_nodes/Adan-GPT-Vision/requirements.txt
```

numpy、Pillow 由 ComfyUI 提供；额外依赖只有 Brotli。

## RunningHub

需要平台执行环境先安装节点源码及依赖，再导入工作流 JSON。是否支持用户自行提交节点包、Git 仓库或平台收录，以 RunningHub 当前账号权限和平台流程为准。本项目尚未在 RunningHub 实测，不代表已被平台收录。

提交给平台审核安装时可提供仓库地址 `https://github.com/AdanYang/Adan-GPT-Vision`，或 Git 克隆地址 `https://github.com/AdanYang/Adan-GPT-Vision.git`。注册节点类型为 `GPTVisionAnalyze`，依赖清单在仓库根目录的 `requirements.txt`。

## 使用

1. 搜索 `GPTVisionAnalyze` 新建节点。
2. 填写供应商的完整 API 基础地址、准确模型 ID、分析任务和该供应商的密钥。base_url 默认留空。
3. 图片或 Batch Images 接入 `images`，另一批图片可接入 `images_2`。
4. 将 `text` 输出接到文字预览或后续提示词拼接；将 `diagnostics` 接预览检查 `image_count`。

`examples/vision_analysis.json` 为最小示例。导入后须重新选择参考图并填写接口配置，图片文件不随仓库提供。

| 参数 | 说明 |
|---|---|
| api_mode | 选择供应商实际支持的 chat_completions 或 responses |
| token_parameter | 默认 max_completion_tokens；旧兼容接口可按其文档改为 max_tokens；omit 省略长度参数 |
| max_output_tokens | 默认 4096；Responses 模式映射到 max_output_tokens |
| image_detail | auto / high / low |
| max_image_side | 默认 1536，不放大小图；0 保留原尺寸 |
| timeout_seconds | 默认 180 秒；超时不自动重试 |
| refresh_seed | 只控制 ComfyUI 缓存刷新，不发送给远端模型 |

支持填基础地址或完整 `/chat/completions`、`/responses` 地址；完整地址必须与 api_mode 匹配。

密钥可填在 api_key 控件中，或在运行 ComfyUI 的服务器设置 `COMFY_GPT_VISION_API_KEY`。控件内密钥可能随工作流导出，分享前应清空。图片以 Base64 随请求交给你填写的 API 供应商；本节点不使用独立图床。

如果接口仍返回 br，请确认在 ComfyUI 使用的 Python 环境安装了 requirements.txt，并重启。错误响应头或损坏数据仍可能需要接口商修复。

## 测试

```text
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

18 项本机模拟 HTTP 接口测试通过，包括真正的 Brotli 压缩、错误 br 响应头、六张图片完整发送、两种 API 协议和错误停止。测试使用虚构密钥，不调用真实模型；未验证真实供应商权限、工业设计质量、完整 ComfyUI UI 或 RunningHub 执行。

请求结构依据 [OpenAI 官方视觉输入文档](https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat)。第三方接口可能只实现部分兼容能力。
