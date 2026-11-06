"""Locks the fallback normaliser and the alignment logic.

The DeepSeek path is not tested here — it is model output, and it is
validated by comparing its results against these fallback rules.
"""

from src.extract.name_normalizer import (fallback_canonical, _align,
                                         canonical_key)
from src.shared.models import LineItem


def test_qty_prefix_stripped():
    assert fallback_canonical("2 X MILK 2L") == "MILK 2L"


def test_punctuation_removed():
    assert fallback_canonical("F&N MILK, 2L.") == "F&N MILK 2L"


def test_abbreviations_expanded():
    assert fallback_canonical("PKT ROTI") == "PACKET BREAD"
    assert fallback_canonical("BTL AIR") == "BOTTLE WATER"


def test_price_like_name_dropped():
    assert fallback_canonical("12.90") == ""
    assert fallback_canonical("RM 4.50") == ""


def test_empty_input():
    assert fallback_canonical("") == ""
    assert fallback_canonical("   ") == ""


def test_size_preserved_never_merged():
    """Different sizes are different products — never collapse them."""
    assert fallback_canonical("MILK 1L") != fallback_canonical("MILK 2L")


def test_case_and_spacing_normalised():
    assert fallback_canonical("  milk   2l  ") == "MILK 2L"


def test_align_uses_raw_key_when_present():
    pairs = [{"raw": "FN MILK 2L", "canonical": "F&N MILK 2L"},
             {"raw": "MILK 2L F&N", "canonical": "F&N MILK 2L"}]
    m = _align(["FN MILK 2L", "MILK 2L F&N"], pairs)
    assert m["FN MILK 2L"] == "F&N MILK 2L"
    assert m["MILK 2L F&N"] == "F&N MILK 2L"


def test_align_positional_when_no_raw_key():
    m = _align(["MILK 2L F&N", "ROTI"], ["F&N MILK 2L", "BREAD"])
    assert m["MILK 2L F&N"] == "F&N MILK 2L"
    assert m["ROTI"] == "BREAD"


def test_align_handles_short_response():
    """A truncated model reply must not erase names."""
    m = _align(["MILK", "ROTI"], [{"canonical": "MILK"}])
    assert m["MILK"] == "MILK"
    assert m["ROTI"] == "BREAD"          # ← fallback fills the gap


def test_canonical_key_prefers_canonical():
    assert canonical_key(
        LineItem("x", 1, 1.0, 1.0, canonical="MILK 2L")) == "MILK 2L"
    assert canonical_key(LineItem("PKT ROTI", 1, 1.0, 1.0)) == "PACKET BREAD"
