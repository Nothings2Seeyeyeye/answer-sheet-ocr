"""纯文本清洗工具函数。

仅依赖标准库，可独立单元测试，供识别管线各环节复用。
"""
from __future__ import annotations

import re
from typing import Any


def clean_digits(text: Any) -> str:
    """仅保留数字字符（用于学号等纯数字字段）。"""
    return "".join(ch for ch in str(text) if ch.isdigit())


def clean_chinese_name(text: Any) -> str:
    """仅保留中文字符（CJK 统一表意文字），用于姓名字段。"""
    return re.sub(r"[^一-鿿]", "", str(text))


def clean_number(text: Any) -> str:
    """提取数值字符串：保留数字与小数点，并规范化多余/首尾小数点。"""
    value = "".join(ch for ch in str(text) if ch.isdigit() or ch == ".")
    if value.count(".") > 1:
        first = value.find(".")
        value = value[: first + 1] + value[first + 1 :].replace(".", "")
    if value.startswith("."):
        value = "0" + value
    if value.endswith("."):
        value = value[:-1]
    return value
