"""Review rules shared by the UI, the export functions and the CLI.

One table decides every row's effective status (team decision 13/11):

    | row                | last action | effective status |
    |--------------------|-------------|------------------|
    | no warning         | none        | auto_accepted    |
    | has a warning      | none        | auto_rejected    |
    | any                | Accept      | accepted         |
    | any                | Correct     | corrected (= accepted with a new value) |
    | any                | Reject      | rejected         |

The TOTAL is reviewed like a row. A receipt may be exported only when
EVERY row and the TOTAL are auto_accepted / accepted / corrected. One
rejected or still auto-rejected entry blocks the whole receipt — both
ledger.csv and audit_export.json — even though ledger.csv has no total
column.

Accepting a row never removes its warnings: they travel with the row
(`warnings`) into the audit export so the trace of the original flag
stays visible.
"""

import json

from src.extract.name_normalizer import fallback_canonical

AUTO_ACCEPTED = "auto_accepted"
AUTO_REJECTED = "auto_rejected"
ACCEPTED = "accepted"
CORRECTED = "corrected"
REJECTED = "rejected"

EXPORTABLE = {AUTO_ACCEPTED, ACCEPTED, CORRECTED}

STATUS_LABEL = {
    AUTO_ACCEPTED: "Auto-accepted",
    AUTO_REJECTED: "Auto-rejected (needs review)",
    ACCEPTED: "Accepted",
    CORRECTED: "Corrected",
    REJECTED: "Rejected",
}

_FROM_DECISION = {"accept": ACCEPTED, "correct": CORRECTED, "reject": REJECTED}

# total_check statuses (set by check_total) that leave the TOTAL unverified
UNVERIFIED_TOTAL = {"insufficient", "no_total"}


def _item_key(item):
    """Same key Check 3 pools on (canonical, else normalised name)."""
    return item.get("canonical") or fallback_canonical(item.get("name", ""))


def map_flags(data, result):
    """Row index (or 'total') → list of {type, cell, reason}.

    A row can carry several warnings (e.g. duplicate AND unusual price);
    all of them are kept so the reviewer sees what they are accepting.
    """
    flag_map = {}

    def add(key, ftype, cell, reason):
        flag_map.setdefault(key, []).append(
            {"type": ftype, "cell": cell, "reason": reason})

    items = data.get("items", [])
    for a in result.get("anomalies", []):
        ev = a.get("evidence", {}) or {}
        atype = a["anomaly_type"]
        if atype == "incorrect_total":
            if "line_index" in ev:
                add(ev["line_index"], "incorrect_total", "line", a["reason"])
            else:
                add("total", "incorrect_total", "line", a["reason"])
        elif atype == "duplicate":
            for idx in ev.get("line_indexes", []):
                add(idx, "duplicate", "name", a["reason"])
        elif atype == "unusual_price":
            canonical = ev.get("canonical", "")
            for i, item in enumerate(items):
                if _item_key(item) == canonical:
                    add(i, "unusual_price", "unit", a["reason"])
                    break

    # rows the arithmetic gate sent to review (e.g. low OCR confidence)
    # but that no check flagged: still need a human decision
    for i, item in enumerate(items):
        if item.get("review") == "review_required" and i not in flag_map:
            add(i, "needs_review", "name", item.get("reason", ""))

    # a total the check could NOT verify is not a pass: ask the reviewer
    tc = result.get("total_check") or {}
    if tc.get("status") in UNVERIFIED_TOTAL:
        add("total", "total_unverified", "line", tc.get("reason", ""))
    elif tc.get("status") == "mismatch_explained_by_duplicate":
        # case (a): the printed TOTAL disagrees with the items by exactly
        # the repeated line — the reviewer must decide which one is wrong
        add("total", "total_vs_duplicate", "line", tc.get("reason", ""))

    return flag_map


def row_status(flagged: bool, decision):
    if decision in _FROM_DECISION:
        return _FROM_DECISION[decision]
    return AUTO_REJECTED if flagged else AUTO_ACCEPTED


def label(key, data):
    if key == "total":
        return "TOTAL"
    return f"Row {key}: {data['items'][key].get('name', '')}"


def review_state(data, result, decisions):
    """Every row + TOTAL with its warnings and effective status, plus the
    export gate: {'rows', 'total', 'blockers', 'can_export'}."""
    decisions = decisions or {}
    flags = map_flags(data, result)
    rows = []
    for i, _ in enumerate(data.get("items", [])):
        f = flags.get(i, [])
        d = decisions.get(str(i))
        rows.append({"key": i, "label": label(i, data), "flags": f,
                     "decision": d, "status": row_status(bool(f), d)})
    f = flags.get("total", [])
    d = decisions.get("total")
    total = {"key": "total", "label": "TOTAL", "flags": f, "decision": d,
             "status": row_status(bool(f), d)}
    blockers = [r for r in rows + [total] if r["status"] not in EXPORTABLE]
    return {"rows": rows, "total": total, "blockers": blockers,
            "can_export": not blockers}


def apply_review(data, result, decisions):
    """Copy of the receipt stamped with the review outcome.

    item['review']   = effective status (table above)
    item['warnings'] = the original warning reasons, kept after Accept
    total_review / total_warnings likewise for the TOTAL
    _review          = the decisions + gate, so export can re-check them
    """
    out = json.loads(json.dumps(data))
    state = review_state(out, result, decisions)
    for row, item in zip(state["rows"], out.get("items", [])):
        if "review" in item and item["review"] not in STATUS_LABEL:
            item["extractor_review"] = item["review"]
        item["review"] = row["status"]
        item["warnings"] = [f"{f['type']}: {f['reason']}" for f in row["flags"]]
    out["total_review"] = state["total"]["status"]
    out["total_warnings"] = [f"{f['type']}: {f['reason']}"
                             for f in state["total"]["flags"]]
    out["_review"] = {
        "decisions": dict(decisions or {}),
        "can_export": state["can_export"],
        "blockers": [{"entry": b["label"], "status": b["status"]}
                     for b in state["blockers"]],
    }
    return out


def blocker_text(blockers) -> str:
    """'Row 2: EGG (auto-rejected …: duplicate); TOTAL (rejected)'."""
    parts = []
    for b in blockers:
        why = ", ".join(sorted({f["type"] for f in b.get("flags", [])}))
        st = STATUS_LABEL.get(b["status"], b["status"])
        parts.append(f"{b['label']} ({st}{': ' + why if why else ''})")
    return "; ".join(parts)
