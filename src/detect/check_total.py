"""Check 1: Incorrect total. DETERMINISTIC — no model.

    E = S − D + T        flag if |E − X| > ε     ε = max(0.01, 0.001 × E)

Worked example (must pass in tests):
    items 6 + 5 + 4 = 15;  E = 15 − 2 + 1 = 14
    observed 16;  difference 2 > ε  →  mismatch
"""

from src.shared.arithmetic import receipt_ok, epsilon, is_vacuous
from src.shared.models import Anomaly
from src.explain.templates import render


def _check_printed_subtotal(receipt) -> list:
    """Printed SUBTOTAL / TOTAL QTY amount vs the sum of the item lines.

    Catches an altered subtotal even when the final total is right
    (demo image test_image_incorrect_total.jpg: items sum to 327.00,
    the 'Total Qty' row prints 377.00).
    """
    printed = getattr(receipt, "printed_subtotal", None)
    if printed is None:
        return []
    items_sum = receipt.subtotal()
    if is_vacuous(items_sum, printed):
        return []
    ratio = items_sum / printed
    if ratio < 0.8 or ratio > 1.2:
        return []          # parser probably missed rows — unreliable
    ok, expected, diff = receipt_ok(items_sum, 0.0, 0.0, printed)
    if ok:
        return []
    direction = "higher" if diff > 0 else "lower"
    facts = {"expected": expected, "observed": printed,
             "diff": abs(diff), "direction": direction}
    return [Anomaly(
        anomaly_type="incorrect_total",
        reason=render("incorrect_total_subtotal", facts),
        evidence={"region": receipt.provenance.get("_subtotal_region",
                                                   "receipt_subtotal"),
                  "ref_value": expected, "observed": printed,
                  "difference": diff, "tolerance": epsilon(expected),
                  "level": "subtotal", "_facts": facts},
        confidence=min(1.0, abs(diff) / max(expected, 0.01)),
    )]


def _set_status(receipt, status, reason="", **extra):
    """Record what the receipt-level check actually did.

    A skipped check is NOT a pass. The UI asks the reviewer to look at
    the TOTAL when the status is 'insufficient' or 'no_total'.
    """
    receipt.provenance["_total_check"] = {"status": status,
                                          "reason": reason, **extra}


def check(receipt, discount: float = 0.0) -> list:
    anomalies = _check_printed_subtotal(receipt)
    if receipt.receipt_total is None:
        _set_status(receipt, "no_total",
                    "No printed TOTAL was read, so the total could not be "
                    "checked.")
        return anomalies

    subtotal = receipt.subtotal()
    tax = receipt.tax or 0.0

    # ── Guard 1: vacuous case ───────────────────────────────
    if is_vacuous(subtotal, receipt.receipt_total):
        _set_status(receipt, "insufficient",
                    "Not enough data to check the total: no line-item "
                    "amounts were read." if subtotal == 0.0 else
                    "Not enough data to check the total: the printed "
                    "total is 0.", items_sum=subtotal,
                    observed=receipt.receipt_total)
        return anomalies

    # ── Guard 2: coverage check ─────────────────────────────
    # If extracted items sum to far less (or more) than the receipt
    # total, the parser missed items or produced bogus prices.
    # The receipt-level check is unreliable in this case — skip it
    # and fall through to the line-level check only.
    if receipt.receipt_total > 0:
        ratio = subtotal / receipt.receipt_total
        if ratio < 0.8 or ratio > 1.2:
            # skip the receipt-level verdict, fall through to line-level.
            # The observed gap is KEPT: Detector.run may still explain it
            # by a repeated line (see coverage_followup); otherwise the
            # TOTAL goes to review as 'insufficient'.
            gap = round(receipt.receipt_total - (subtotal - discount + tax), 2)
            unread = len(receipt.provenance.get("_unparsed_rows", []))
            _set_status(
                receipt, "insufficient",
                f"Not enough data to check the total: line items read sum "
                f"to RM {subtotal:.2f}, {ratio:.0%} of the printed total "
                f"RM {receipt.receipt_total:.2f} (outside 80–120%), so OCR "
                f"may have missed or merged rows"
                + (f" ({unread} unreadable item row(s))" if unread else "")
                + f". Observed difference RM {gap:+.2f} — please check "
                f"the image.",
                ratio=round(ratio, 4), items_sum=subtotal, tax=tax,
                discount=discount, observed=receipt.receipt_total,
                difference=gap, unreadable_rows=unread)
        else:
            _set_status(receipt, "checked", ratio=round(ratio, 4))
            # ── Receipt-level check ──────────────────────────
            ok, expected, diff = receipt_ok(
                subtotal, discount, tax, receipt.receipt_total)

            if not ok:
                direction = "higher" if diff > 0 else "lower"
                facts = {"expected": expected, "observed": receipt.receipt_total,
                         "diff": abs(diff), "direction": direction,
                         "subtotal": subtotal, "tax": tax}
                tpl = ("incorrect_total_receipt_tax" if tax
                       else "incorrect_total_receipt")
                anomalies.append(Anomaly(
                    anomaly_type="incorrect_total",
                    reason=render(tpl, facts),
                    evidence={"region": receipt.provenance.get(
                              "_total_region") or "receipt_total",
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


def coverage_followup(receipt, duplicate_flags, discount: float = 0.0) -> list:
    """Re-examine a receipt the coverage guard skipped, using duplicates.

    Case (a) — the gap is explained: the items exceed the total, no item
    row was unreadable, and removing the repeated copy of each duplicated
    line makes S − D + T match the printed total (demo image: Plastic
    31.00 + Plastie 31.00 = 62.00 vs TOTAL 31.00). The observed gap is
    attached to the DUPLICATE warning(s) and the total check is marked
    'mismatch_explained_by_duplicate', so the UI puts the TOTAL under
    review with the arithmetic. No extra incorrect_total anomaly is
    emitted: the one fault already flagged (the repeat) explains the gap,
    and the labelled error set counts it once, as a duplicate.

    Case (b) — anything else stays 'insufficient': the TOTAL is not
    reported as wrong, but it is not passed either; the reviewer sees the
    observed difference and the reason.

    Mutates `duplicate_flags` in place; returns [] (kept as a list so the
    Detector can add its result like any other check).
    """
    tc = receipt.provenance.get("_total_check") or {}
    if tc.get("status") != "insufficient" or "ratio" not in tc:
        return []
    if tc["ratio"] <= 1.2 or tc.get("unreadable_rows"):
        return []
    extra, dup_rows, names, used = 0.0, [], [], []
    for f in duplicate_flags:
        idxs = f.evidence.get("line_indexes") or []
        if len(idxs) != 2:
            continue
        extra += receipt.items[idxs[1]].line_total
        dup_rows += idxs
        names.append(receipt.items[idxs[0]].name)
        used.append(f)
    if not used:
        return []
    subtotal = receipt.subtotal()
    tax = receipt.tax or 0.0
    ok, _, _ = receipt_ok(round(subtotal - extra, 2), discount, tax,
                          receipt.receipt_total)
    if not ok:
        return []
    expected = round(subtotal - discount + tax, 2)
    diff = round(receipt.receipt_total - expected, 2)
    facts = {"items_sum": subtotal, "observed": receipt.receipt_total,
             "diff": abs(diff), "dup_amount": round(extra, 2),
             "names": ", ".join(f"'{n}'" for n in names)}
    reason = render("incorrect_total_duplicate", facts)
    gap = {"items_sum": subtotal, "observed_total": receipt.receipt_total,
           "difference": diff, "duplicate_amount": round(extra, 2)}
    for f in used:
        f.reason = f.reason + " " + reason
        f.evidence["total_gap"] = gap
        f.evidence["_facts"] = {**f.evidence.get("_facts", {}), **facts}
    _set_status(receipt, "mismatch_explained_by_duplicate", reason,
                items_sum=subtotal, observed=receipt.receipt_total,
                difference=diff, duplicate_rows=sorted(set(dup_rows)))
    return []
