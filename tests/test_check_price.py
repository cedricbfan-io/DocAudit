"""Locks the MAD = 0 guard, the pool guards, and the pool builder."""

from src.detect.check_price import modified_z, check
from src.detect.reference_builder import (build_pool, pool_stats,
                                          degenerate_items)
from src.shared.models import Receipt, LineItem


def _r(items, currency="MYR", rid="t"):
    r = Receipt(vendor="X", date="2026-10-20", receipt_id=rid,
                currency=currency)
    r.items = items
    return r


def test_mad_zero_returns_none():
    """All prices identical → MAD = 0 → must not divide by zero."""
    assert modified_z(5.0, [5.0, 5.0, 5.0, 5.0, 5.0]) is None


def test_normal_price_not_flagged():
    prices = [3.40, 3.50, 3.60, 3.45, 3.55, 3.50]
    z = modified_z(3.50, prices)
    assert z is not None and abs(z) <= 3.5


def test_outlier_flagged():
    prices = [3.40, 3.50, 3.60, 3.45, 3.55, 3.50]
    z = modified_z(30.00, prices)
    assert z is not None and abs(z) > 3.5


def test_below_min_comparables_skipped():
    r = _r([LineItem("Milk", 1, 99.0, 99.0)])
    assert check(r, {"MILK": [3.5, 3.6]}) == []


def test_pool_builder_normalises_names():
    """Pool keys must be canonical — otherwise it fragments."""
    r = _r([LineItem("PKT ROTI", 1, 3.5, 3.5, canonical="PACKET BREAD")])
    pool = build_pool([r], min_obs=1)
    assert "PACKET BREAD" in pool
    assert "PKT ROTI" not in pool


def test_pool_rejects_non_myr():
    r = _r([LineItem("Milk", 1, 3.5, 3.5, canonical="MILK")],
           currency="USD")
    assert build_pool([r], min_obs=1) == {}


def test_pool_respects_min_observations():
    r = _r([LineItem("Milk", 1, 3.5, 3.5, canonical="MILK")])
    assert build_pool([r], min_obs=5) == {}
    assert build_pool([r], min_obs=1) == {"MILK": [3.5]}


def test_degenerate_items_detected():
    pool = {"MILK": [3.5, 3.5, 3.5], "EGG": [1.0, 1.2, 1.4]}
    assert degenerate_items(pool) == ["MILK"]


def test_pool_stats_warns_when_empty():
    assert "warning" in pool_stats({})
