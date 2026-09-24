"""PaddleOCR-VL 在线 API 封装（可选高精度后端）。

使用 PaddleOCR-VL 在线服务识别图片中的文字，返回 markdown 文本。
启用前需设置环境变量 ``PADDLEOCR_API_TOKEN``：

    export PADDLEOCR_API_TOKEN="your-token"

未设置 TOKEN、未安装 ``requests``、超过 ``PADDLEOCR_API_TIMEOUT``（默认 180 秒）、
或服务不可用时，:func:`recognize_text` 返回空字符串，调用方应据此降级到本地识别后端。

注意：TOKEN 属于敏感凭据，切勿硬编码进代码或提交到仓库。
"""
from __future__ import annotations

import json
import logging
import os
import time

try:
    import requests
except ImportError:  # requests 为可选依赖
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL = "PaddleOCR-VL-1.6"
_POLL_INTERVAL = 3.0
_DEFAULT_TIMEOUT = 180.0  # 默认总超时（秒），可用环境变量 PADDLEOCR_API_TIMEOUT 覆盖


def _token() -> str:
    return os.environ.get("PADDLEOCR_API_TOKEN", "")


def _timeout() -> float:
    try:
        return float(os.environ.get("PADDLEOCR_API_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    except ValueError:
        return _DEFAULT_TIMEOUT


def _optional_payload() -> str:
    return json.dumps(
        {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
    )


def recognize_text(image_bytes: bytes, filename: str = "roi.jpg") -> str:
    """提交图片到在线 OCR 服务，轮询并返回 markdown 文本。

    Args:
        image_bytes: 图片二进制内容（JPEG）。
        filename: 提交时的文件名。

    Returns:
        识别出的 markdown 文本；失败时返回空字符串。
    """
    if requests is None or not _token():
        return ""
    headers = {"Authorization": f"bearer {_token()}"}
    data = {"model": MODEL, "optionalPayload": _optional_payload()}
    files = {"file": (filename, image_bytes, "image/jpeg")}

    try:
        resp = requests.post(JOB_URL, headers=headers, data=data, files=files, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["data"]["jobId"]
    except Exception:
        logger.warning("在线 PaddleOCR API 提交任务失败")
        return ""

    deadline = time.time() + _timeout()
    jsonl_url = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=30)
            r.raise_for_status()
            payload = r.json()["data"]
        except Exception:
            logger.warning("在线 PaddleOCR API 查询任务状态失败")
            return ""
        state = payload.get("state")
        if state == "done":
            jsonl_url = payload["resultUrl"]["jsonUrl"]
            break
        if state == "failed":
            logger.warning("在线 PaddleOCR API 任务失败: %s", payload.get("errorMsg", ""))
            return ""
        time.sleep(_POLL_INTERVAL)

    if not jsonl_url:
        logger.warning("在线 PaddleOCR API 超时（超过 %.0fs）", _timeout())
        return ""

    try:
        jr = requests.get(jsonl_url, timeout=30)
        jr.raise_for_status()
    except Exception:
        logger.warning("在线 PaddleOCR API 下载结果失败")
        return ""

    texts: list[str] = []
    for line in jr.text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)["result"]
        except Exception:
            continue
        for res in result.get("layoutParsingResults", []):
            texts.append(res.get("markdown", {}).get("text", ""))
    return "\n".join(texts)
