"""The ε tolerance — defined ONCE, shared by extraction and Check 1.

This single function powers the arithmetic gate, the biggest lever on
human workload: a mis-mapped axis breaks the arithmetic, so the failure
announces itself.
"""

from config.thresholds_loader import THRESHOLDS

_REL = THRESHOLDS["total"]["tolerance_relative"]
_FLOOR = THRESHOLDS["total"]["tolerance_floor"]
_R_FLOOR = THRESHOLDS["total"].get("receipt_tolerance_floor", _FLOOR)


def epsilon(expected: float) -> float:
    """ε = max(0.01, 0.001 × expected)"""
    return max(_FLOOR, _REL * abs(expected))


def receipt_epsilon(expected: float) -> float:
    """Receipt-level ε — looser to tolerate OCR misses."""
    return max(_R_FLOOR, _REL * abs(expected))


def line_ok(qty: float, unit_price: float, line_total: float) -> bool:
    """Line level: |line_total − qty × unit_price| ≤ ε"""
    expected = qty * unit_price
    return abs(expected - line_total) <= epsilon(expected)


def receipt_ok(subtotal: float, discount: float,
               tax: float, printed_total: float) -> tuple:
    """Receipt level. Returns (ok, expected, difference).

        E = S − D + T        flag if |E − X| > ε

    If the schema carries no discount field, pass discount=0.0 and the
    formula degrades to subtotal + tax.
    """
    expected = round(subtotal - discount + tax, 2)
    difference = round(printed_total - expected, 2)
    return abs(difference) <= receipt_epsilon(expected), expected, difference


def is_vacuous(subtotal: float, printed_total: float) -> bool:
    """Guard: if either side is zero the comparison is meaningless."""
    return subtotal == 0.0 or printed_total == 0.0
