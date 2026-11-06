"""Check 1: Incorrect total. DETERMINISTIC — no model.

    E = S − D + T        flag if |E − X| > ε     ε = max(0.01, 0.001 × E)

Worked example (must pass in tests):
    items 6 + 5 + 4 = 15;  E = 15 − 2 + 1 = 14
    observed 16;  difference 2 > ε  →  mismatch
"""

from src.shared.arithmetic import receipt_ok, epsilon, is_vacuous
from src.shared.models import Anomaly
from src.explain.templates import render


def check(receipt, discount: float = 0.0) -> list:
    anomalies = []
    if receipt.receipt_total is None:
        return anomalies

    subtotal = receipt.subtotal()
    tax = receipt.tax or 0.0

    # ── Guard 1: vacuous case ───────────────────────────────
    if is_vacuous(subtotal, receipt.receipt_total):
        return anomalies

    # ── Guard 2: discount absence ───────────────────────────
    # If the schema carries no discount field, E degrades to S + T.
    ok, expected, diff = receipt_ok(
        subtotal, discount, tax, receipt.receipt_total)

    if not ok:
        direction = "higher" if diff > 0 else "lower"
        facts = {"expected": expected, "observed": receipt.receipt_total,
                 "diff": abs(diff), "direction": direction}
        anomalies.append(Anomaly(
            anomaly_type="incorrect_total",
            reason=render("incorrect_total_receipt", facts),
            evidence={"region": "receipt_total",
                      "ref_value": expected,
                      "observed": receipt.receipt_total,
                      "difference": diff,
                      "tolerance": epsilon(expected),
                      "_facts": facts},
            confidence=min(1.0, abs(diff) / max(expected, 0.01)),
        ))

    # ── Line-level check ────────────────────────────────────
    for idx, item in enumerate(receipt.items):
        exp = round(item.qty * item.unit_price, 2)
        if abs(exp - item.line_total) > epsilon(exp):
            facts = {"expected": exp, "observed": item.line_total,
                     "qty": item.qty, "unit": item.unit_price,
                     "name": item.name}
            anomalies.append(Anomaly(
                anomaly_type="incorrect_total",
                reason=render("incorrect_total_line", facts),
                evidence={"region": item.region, "ref_value": exp,
                          "observed": item.line_total,
                          "line_index": idx,
                          "ocr_confidence": item.ocr_confidence,
                          "_facts": facts},
                confidence=0.9,
            ))
    return anomalies
