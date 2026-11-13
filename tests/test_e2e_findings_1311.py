"""Bugs found by the end-to-end run on 13/11 (UI + CLI on non-SROIE receipts)."""

import pytest

from src.extract.fallback_extract import _find_date, _find_vendor, _parse_date
from src.ui import review


@pytest.mark.parametrize("line,want", [
    ("2026-11-1309:10", "2026-11-13"),       # OCR glued the time on (was 1309-11-26)
    ("2026-11-13 09:10", "2026-11-13"),
    ("2026/11/13", "2026-11-13"),
    ("2026年11月13日 12:30", "2026-11-13"),
    ("25/12/2018 10:02", "2018-12-25"),       # SROIE day-first stays
    ("Date: 03-01-18", "2018-01-03"),
    ("TEL 03-8920 2133", None),               # a phone number is not a date
])
def test_dates(line, want):
    assert _find_date([line]) == want


def test_parse_date_keeps_unknown_text():
    assert _parse_date("25/12/2018") == "2018-12-25"
    assert _parse_date("Apr 3 2018") == "Apr 3 2018"


def test_short_chinese_shop_name_is_the_vendor():
    regs = [{"text": "便利店"}, {"text": "2026-11-13 09:10"}, {"text": "矿泉水 x2"}]
    assert _find_vendor([], regs) == "便利店"
    regs = [{"text": "2026-11-1309:10"}, {"text": "CORNER MART"}]
    assert _find_vendor([], regs) == "CORNER MART"


def test_unusual_price_on_two_identical_lines_is_not_doubled():
    data = {"items": [{"name": "MILK 1L", "canonical": "MILK 1L"},
                      {"name": "MILK 1L", "canonical": "MILK 1L"},
                      {"name": "EGG", "canonical": "EGG"}]}
    a = {"anomaly_type": "unusual_price", "reason": "high",
         "evidence": {"canonical": "MILK 1L"}}
    flags = review.map_flags(data, {"anomalies": [a, dict(a)]})
    assert [f["type"] for f in flags[0]] == ["unusual_price"]
    assert [f["type"] for f in flags[1]] == ["unusual_price"]
    assert 2 not in flags


def test_ui_and_cli_show_the_receipt_currency():
    from src.ui.app import _sym, _build_html_table
    data = {"currency": "USD", "receipt_total": 23.27, "items": [
        {"name": "BREAD", "canonical": "BREAD", "qty": 1, "unit_price": 3.49,
         "line_total": 3.49}]}
    assert _sym(data) == "$" and _sym({"currency": "CNY"}) == "¥"
    assert _sym({}) == "RM"
    t = _build_html_table(data, {"anomalies": []})
    assert "Unit ($)" in t and "(RM)" not in t
