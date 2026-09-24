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
