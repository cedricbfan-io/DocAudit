"""Export. CONFIRMED ROWS ONLY.

Rule stated in the proposal and enforced here: nothing reaches the
ledger until a human confirms it. Rejected flags never propagate.
"""

import csv
import json
from datetime import datetime
from pathlib import Path


def _confirmed(items: list) -> list:
    """Confirmed = the human accepted the row, OR it was never flagged.

    'corrected' counts as confirmed: the human fixed it, so the fixed
    values are authoritative.
    """
    return [i for i in items
            if i.get("review") in ("unreviewed", "corrected", "accepted")]


_FIELDS = ["receipt_id", "vendor", "date", "currency", "item",
           "canonical", "qty", "unit_price", "line_total", "review"]


def to_ledger_csv(receipts: list, path: str = "ledger.csv") -> str:
    """One row per confirmed line item."""
    rows = []
    for r in receipts:
        for i in _confirmed(r.get("items", [])):
            rows.append({
                "receipt_id": r.get("receipt_id", ""),
                "vendor": r.get("vendor", ""),
                "date": r.get("date", ""),
                "currency": r.get("currency", "MYR"),
                "item": i.get("name", ""),
                "canonical": i.get("canonical", ""),
                "qty": i.get("qty", ""),
                "unit_price": i.get("unit_price", ""),
                "line_total": i.get("line_total", ""),
                "review": i.get("review", ""),
            })

    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


def to_audit_json(receipts: list, results: list,
                  path: str = "audit_export.json") -> str:
    """Full audit trail: confirmed rows, plus every flag with its reason.

    This is the file you show when asked "where did this number come
    from?" — evidence regions travel with each flag.
    """
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "currency": "MYR",
        "receipts": [],
    }
    by_id = {r.get("receipt_id"): r for r in receipts}

    for res in results:
        rid = res.get("receipt_id")
        src = by_id.get(rid, {})
        payload["receipts"].append({
            "receipt_id": rid,
            "vendor": src.get("vendor"),
            "date": src.get("date"),
            "receipt_total": src.get("receipt_total"),
            "confirmed_items": _confirmed(src.get("items", [])),
            "flagged_items": [i for i in src.get("items", [])
                              if i.get("review") == "review_required"],
            "anomalies": res.get("anomalies", []),
            "summary": res.get("summary"),
        })

    p = Path(path)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def export_summary(receipts: list, results: list) -> dict:
    """Counts for the UI — how much was auto-accepted vs reviewed."""
    total = flagged = confirmed = rejected = 0
    for r in receipts:
        for i in r.get("items", []):
            total += 1
            if i.get("review") == "review_required":
                flagged += 1
            elif i.get("review") in ("unreviewed", "corrected", "accepted"):
                confirmed += 1
            else:
                rejected += 1
    return {
        "total_items": total,
        "flagged": flagged,
        "confirmed": confirmed,
        "rejected": rejected,
        "review_load": round(flagged / total, 4) if total else 0.0,
        "anomalies": sum(len(r.get("anomalies", [])) for r in results),
    }
