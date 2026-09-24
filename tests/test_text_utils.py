"""text_utils 纯函数单元测试（零第三方依赖）。"""
from __future__ import annotations

import text_utils


def test_clean_digits_keeps_only_digits():
    assert text_utils.clean_digits("2021162992") == "2021162992"
    assert text_utils.clean_digits("学号2021") == "2021"
    assert text_utils.clean_digits(123) == "123"
    assert text_utils.clean_digits("12.34") == "1234"
    assert text_utils.clean_digits("abc") == ""


def test_clean_chinese_name_keeps_only_cjk():
    assert text_utils.clean_chinese_name("李明") == "李明"
    assert text_utils.clean_chinese_name("李 明") == "李明"
    assert text_utils.clean_chinese_name("张三abc123") == "张三"
    assert text_utils.clean_chinese_name("abc") == ""
    assert text_utils.clean_chinese_name("") == ""


def test_clean_number_normalizes_decimal_points():
    assert text_utils.clean_number("98") == "98"
    assert text_utils.clean_number("12.5") == "12.5"
    assert text_utils.clean_number(".5") == "0.5"
    assert text_utils.clean_number("5.") == "5"
    assert text_utils.clean_number("12.34.5") == "12.345"
    assert text_utils.clean_number("1.2.3") == "1.23"
    assert text_utils.clean_number("abc") == ""


def test_apply_total_check_corrects_wrong_total():
    scores = [
        {"field": "score_1", "label": "第1题", "value": "20", "confidence": 0.94},
        {"field": "total", "label": "总分", "value": "99", "confidence": 0.92},
    ]
    text_utils.apply_total_check(scores)
    total = next(s for s in scores if s["field"] == "total")
    assert total["value"] == "20"
    assert total["confidence"] == 0.0


def test_apply_total_check_fills_missing_total():
    scores = [
        {"field": "score_1", "label": "第1题", "value": "20", "confidence": 0.94},
        {"field": "score_2_1", "label": "第2题(1)", "value": "8", "confidence": 0.90},
        {"field": "total", "label": "总分", "value": "", "confidence": 0.0},
    ]
    text_utils.apply_total_check(scores)
    total = next(s for s in scores if s["field"] == "total")
    assert total["value"] == "28"


def test_apply_total_check_keeps_correct_total():
    scores = [
        {"field": "score_1", "label": "第1题", "value": "20", "confidence": 0.94},
        {"field": "total", "label": "总分", "value": "20", "confidence": 0.92},
    ]
    text_utils.apply_total_check(scores)
    total = next(s for s in scores if s["field"] == "total")
    assert total["value"] == "20"
    assert total["confidence"] == 0.92  # 未修改


def test_apply_total_check_skips_when_sub_empty():
    scores = [
        {"field": "score_1", "label": "第1题", "value": "", "confidence": 0.0},
        {"field": "total", "label": "总分", "value": "99", "confidence": 0.92},
    ]
    text_utils.apply_total_check(scores)
    total = next(s for s in scores if s["field"] == "total")
    assert total["value"] == "99"  # 小题为空，不修正


def test_apply_total_check_formats_decimal():
    scores = [
        {"field": "score_1", "label": "第1题", "value": "20.5", "confidence": 0.94},
        {"field": "total", "label": "总分", "value": "30", "confidence": 0.92},
    ]
    text_utils.apply_total_check(scores)
    total = next(s for s in scores if s["field"] == "total")
    assert total["value"] == "20.5"
