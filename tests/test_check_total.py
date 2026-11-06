"""Locks the worked example and both guards."""

from src.shared.arithmetic import receipt_ok, line_ok, is_vacuous, epsilon
from src.shared.models import Receipt, LineItem
from src.detect import check_total


def _receipt(total, items):
    r = Receipt(vendor="Tesco", date="2026-10-20", receipt_total=total,
                tax=1.0, receipt_id="t1")
    r.items = items
    return r


def test_worked_example_mismatch():
    items = [LineItem("Apple", 2, 3.0, 6.0),
             LineItem("Milk", 1, 5.0, 5.0),
             LineItem("Bread", 1, 4.0, 4.0)]
    r = _receipt(16.0, items)                      # printed 16, expected 14
    ok, expected, diff = receipt_ok(15.0, 2.0, 1.0, 16.0)
    assert not ok and expected == 14.0 and diff == 2.0
    flags = check_total.check(r, discount=2.0)
    assert any(f.anomaly_type == "incorrect_total" for f in flags)


def test_worked_example_correct():
    ok, expected, _ = receipt_ok(15.0, 2.0, 1.0, 14.0)
    assert ok and expected == 14.0


def test_epsilon_boundary():
    assert epsilon(14.0) == 0.014                  # 0.001 × 14 > 0.01
    assert epsilon(1.0) == 0.01                    # floor wins


def test_vacuous_guard():
    assert is_vacuous(0.0, 16.0)
    assert check_total.check(_receipt(16.0, [])) == []


def test_line_guard():
    assert line_ok(2, 3.50, 7.00)
    assert not line_ok(2, 3.50, 7.50)


def test_line_level_flag_fires():
    r = _receipt(8.00, [LineItem("Milk", 2, 3.50, 8.00)])   # 7.00 ≠ 8.00
    flags = check_total.check(r)
    assert any(f.evidence.get("line_index") == 0 for f in flags)
