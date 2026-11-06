"""Locks the three-way split and within-receipt detection."""

from src.detect.check_duplicate import (check_within, classify_pair,
                                        MATCH, NON_MATCH)
from src.shared.models import Receipt, LineItem


def _r(items, vendor="Tesco", total=None, rid="a"):
    r = Receipt(vendor=vendor, date="2026-10-20", receipt_id=rid)
    r.items = items
    r.receipt_total = total if total is not None else r.subtotal()
    return r


def test_within_receipt_duplicate_detected():
    r = _r([LineItem("Milk", 1, 3.50, 3.50),
            LineItem("Milk", 1, 3.50, 3.50)])
    flags = check_within(r)
    assert len(flags) == 1
    assert flags[0].anomaly_type == "duplicate"


def test_different_qty_not_duplicate():
    r = _r([LineItem("Milk", 1, 3.50, 3.50),
            LineItem("Milk", 2, 3.50, 7.00)])
    assert check_within(r) == []


def test_identical_receipts_score_match():
    a = _r([LineItem("Milk", 1, 3.50, 3.50)], total=3.50, rid="a")
    b = _r([LineItem("Milk", 1, 3.50, 3.50)], total=3.50, rid="b")
    verdict, score = classify_pair(a, b)
    assert verdict == MATCH and score >= 0.9


def test_different_receipts_non_match():
    a = _r([LineItem("Milk", 1, 3.50, 3.50)], vendor="Tesco",
           total=3.50, rid="a")
    b = _r([LineItem("Rice", 1, 25.00, 25.00)], vendor="Mydin",
           total=25.00, rid="b")
    verdict, _ = classify_pair(a, b)
    assert verdict == NON_MATCH
