"""Export. WHOLE RECEIPTS ONLY, and only after review.

Rule (team decision 13/11, see src/ui/review.py): a receipt reaches
ledger.csv / audit_export.json only when every line item AND its TOTAL
are auto-accepted, accepted or corrected. One rejected (or still
auto-rejected) entry blocks the receipt in BOTH files. The check runs
here, before anything is written, so it also holds when these functions
are called directly or from the CLI — not only from the UI.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from src.ui import review


class ExportBlocked(Exception):
    """Raised before writing when a receipt is not fully accepted."""

    def __init__(self, blockers):
        self.blockers = blockers
        super().__init__("Export blocked: " + "; ".join(
            f"{b['receipt_id']} · {b['entry']} ({b['status']})"
            for b in blockers))


def _result_for(receipt, results):
    if not results:
        return None
    rid = receipt.get("receipt_id")
    for res in results:
        if res.get("receipt_id") == rid:
            return res
    return None


def check_exportable(receipts: list, results: list = None) -> list:
    """Blockers for these receipts ([] = all may be exported).

    With the detector result the gate is recomputed from the stored
    decisions; without it, only a receipt stamped by review.apply_review
    can pass (an unreviewed receipt is never exported).
    """
    blockers = []
    for r in receipts:
        rid = r.get("receipt_id", "")
        stamp = r.get("_review")
        res = _result_for(r, results)
        if res is not None:
            decisions = (stamp or {}).get("decisions", {})
            state = review.review_state(r, res, decisions)
            blockers += [{"receipt_id": rid, "entry": b["label"],
                          "status": b["status"]} for b in state["blockers"]]
        elif stamp is None:
            blockers.append({"receipt_id": rid, "entry": "receipt",
                             "status": "not reviewed"})
        else:
            for i, item in enumerate(r.get("items", [])):
                if item.get("review") not in review.EXPORTABLE:
                    blockers.append({"receipt_id": rid,
                                     "entry": review.label(i, r),
                                     "status": item.get("review")})
            if r.get("total_review") not in review.EXPORTABLE:
                blockers.append({"receipt_id": rid, "entry": "TOTAL",
                                 "status": r.get("total_review")})
    return blockers


def _require_exportable(receipts, results):
    blockers = check_exportable(receipts, results)
    if blockers:
        raise ExportBlocked(blockers)


_FIELDS = ["receipt_id", "vendor", "date", "currency", "item",
           "canonical", "qty", "unit_price", "line_total", "review",
           "warnings"]


def to_ledger_csv(receipts: list, path: str = "ledger.csv",
                  results: list = None) -> str:
    """One row per line item of each fully accepted receipt.

    Raises ExportBlocked (and writes nothing) otherwise.
    """
    _require_exportable(receipts, results)
    rows = []
    for r in receipts:
        for i in r.get("items", []):
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
                # an accepted warning stays visible in the ledger too
                "warnings": " | ".join(i.get("warnings", [])),
            })

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


def to_audit_json(receipts: list, results: list,
                  path: str = "audit_export.json") -> str:
    """Full audit trail of each fully accepted receipt: every row with its
    review status and original warnings, the TOTAL decision, corrections,
    and every flag with its evidence region.

    Raises ExportBlocked (and writes nothing) if any receipt is blocked.
    """
    _require_exportable(receipts, results)
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
            "total_review": src.get("total_review"),
            "total_warnings": src.get("total_warnings", []),
            "total_check": res.get("total_check"),
            "decisions": (src.get("_review") or {}).get("decisions", {}),
            "corrections": src.get("_corrections", []),
            "items": src.get("items", []),
            "anomalies": res.get("anomalies", []),
            "summary": res.get("summary"),
        })

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def export_summary(receipts: list, results: list) -> dict:
    """Counts per effective status (rows only; TOTAL reported apart)."""
    counts = {s: 0 for s in review.STATUS_LABEL}
    total = 0
    for r in receipts:
        for i in r.get("items", []):
            total += 1
            st = i.get("review")
            if st in counts:
                counts[st] += 1
    flagged = counts[review.AUTO_REJECTED]
    return {
        "total_items": total,
        **counts,
        "review_load": round(flagged / total, 4) if total else 0.0,
        "anomalies": sum(len(r.get("anomalies", [])) for r in results),
    }


def export_receipt(data, result, decisions, out_dir) -> tuple:
    """Stamp, gate and write one reviewed receipt into `out_dir`.

    Returns (csv_path, json_path, stamped receipt). Raises ExportBlocked
    before writing.
    Used by the UI and the CLI, so both apply the same rule.
    """
    stamped = review.apply_review(data, result, decisions)
    _require_exportable([stamped], [result])
    out = Path(out_dir)
    csv_path = to_ledger_csv([stamped], str(out / "ledger.csv"), [result])
    json_path = to_audit_json([stamped], [result],
                              str(out / "audit_export.json"))
    return csv_path, json_path, stamped
