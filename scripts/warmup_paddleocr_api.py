"""预热在线 PaddleOCR API（可选）。

首次调用在线 API 会触发服务端冷启动（实测可达 2 分钟以上），
本脚本提前提交一次请求以触发模型加载，之后识别将显著加快。

用法：

    export PADDLEOCR_API_TOKEN="your-token"
    python scripts/warmup_paddleocr_api.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paddleocr_api import recognize_text  # noqa: E402

# 1x1 白色 PNG，仅用于触发服务端模型加载
_MIN_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def main() -> None:
    if not os.environ.get("PADDLEOCR_API_TOKEN"):
        print("未设置环境变量 PADDLEOCR_API_TOKEN，跳过预热。")
        return
    print("正在预热在线 PaddleOCR API（首次可能耗时较长）...")
    recognize_text(_MIN_PNG, filename="warmup.png")
    print("预热完成。")


if __name__ == "__main__":
    main()
