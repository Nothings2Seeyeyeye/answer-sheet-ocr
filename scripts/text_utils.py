"""识别管线纯逻辑工具函数。

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


def apply_total_check(scores: list[dict[str, Any]]) -> None:
    """用「总分 = Σ小题」约束校验并修正总分字段（就地修改）。

    当总分缺失或与各小题之和不一致时，用小题之和覆盖总分，
    并将置信度置 0 以提示人工核对；任一小题为空或不可解析时放弃校验。
    """
    sub_sum = 0.0
    total_item = None
    for item in scores:
        if item["field"] == "total":
            total_item = item
            continue
        value = str(item.get("value", "") or "").strip()
        if value == "":
            return
        try:
            sub_sum += float(value)
        except ValueError:
            return
    if total_item is None:
        return
    total_raw = str(total_item.get("value", "") or "").strip()
    try:
        total_value = float(total_raw) if total_raw else None
    except ValueError:
        total_value = None
    if total_value is None or abs(total_value - sub_sum) > 1e-6:
        total_item["value"] = str(int(sub_sum)) if sub_sum.is_integer() else str(sub_sum)
        total_item["confidence"] = 0.0
