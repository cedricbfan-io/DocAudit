"""Locks the four-field contract and the number guard."""

from src.explain.explain import enforce_contract
from src.explain.number_guard import numbers_ok, guard
from src.shared.models import Anomaly


def test_contract_complete():
    a = Anomaly("incorrect_total", "Total RM 16.00 is high.", {}, 0.9)
    out = enforce_contract(a)
    assert set(out) == {"anomaly_type", "reason", "evidence", "confidence"}
    assert 0.0 <= out["confidence"] <= 1.0


def test_bad_enum_rejected():
    try:
        Anomaly("fraud", "x", {}, 0.5)
        assert False, "should have raised"
    except ValueError:
        pass


def test_number_guard_accepts_correct():
    facts = {"expected": 14.00, "observed": 16.00, "diff": 2.00}
    text = "Total RM 16.00 is RM 2.00 higher than the expected RM 14.00."
    assert numbers_ok(text, facts)


def test_number_guard_rejects_altered():
    facts = {"expected": 14.00, "observed": 16.00, "diff": 2.00}
    text = "Total RM 16.00 is RM 2.50 higher than the expected RM 14.00."
    assert not numbers_ok(text, facts)
    assert guard("fallback", text, facts) == "fallback"


def test_number_guard_rejects_invented():
    facts = {"expected": 14.00, "observed": 16.00, "diff": 2.00}
    text = "Total RM 99.99 is wrong."
    assert not numbers_ok(text, facts)
