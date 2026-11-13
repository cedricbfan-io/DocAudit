"""Review + export from the terminal, with the SAME rules as the web UI.

    python -m src.ui.cli demo/test_image_clean.png
    python -m src.ui.cli demo/demo_multi_fault.json --accept 0,2,3 --accept-total
    python -m src.ui.cli demo/demo_clean.json --reject-total        # → blocked

Prints every row with its status and warnings, then exports to
exports/<receipt>_<time>/ only if every row and the TOTAL are accepted.
Exit code 0 = exported, 2 = blocked (nothing written), 1 = input error.
The old extractor CLI (python -m src.extract.extract) is unchanged.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ui import export, review  # noqa: E402
from src.explain.templates import SYMBOL as _SYMBOL  # noqa: E402


def _load(path):
    """(data, result) through the same code paths as the UI tabs."""
    from src.ui import app
    if path.lower().endswith(".json"):
        out = app.process_json(path)
    else:
        out = app.process(path)
    if app._STATE["receipt"] is None:
        raise SystemExit(f"error: {out[0]}")
    return app._STATE["receipt"], app._STATE["result"]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="receipt image or receipt JSON")
    ap.add_argument("--accept", default="", help="row numbers, e.g. 0,2")
    ap.add_argument("--reject", default="", help="row numbers, e.g. 1")
    ap.add_argument("--correct", action="append", default=[],
                    help="ROW=VALUE (line total) or T=VALUE (TOTAL)")
    ap.add_argument("--accept-total", action="store_true")
    ap.add_argument("--reject-total", action="store_true")
    ap.add_argument("--accept-all", action="store_true",
                    help="accept every flagged entry not decided otherwise")
    ap.add_argument("--out", default="exports")
    a = ap.parse_args(argv)

    if not Path(a.path).exists():
        print(f"error: no such file {a.path}", file=sys.stderr)
        return 1
    data, result = _load(a.path)
    decisions = {}
    for i in _ints(a.accept):
        decisions[str(i)] = "accept"
    for i in _ints(a.reject):
        decisions[str(i)] = "reject"
    for c in a.correct:
        key, val = c.split("=", 1)
        key = "total" if key.upper() == "T" else str(int(key))
        val = round(float(val), 2)
        trace = [f"{f['type']}: {f['reason']}" for f in
                 review.map_flags(data, result).get(
                     "total" if key == "total" else int(key), [])]
        if key == "total":
            data.setdefault("_corrections", []).append(
                {"field": "receipt_total", "from": data.get("receipt_total"),
                 "to": val, "warnings": trace})
            data["receipt_total"] = val
        else:
            item = data["items"][int(key)]
            data.setdefault("_corrections", []).append(
                {"row": int(key), "field": "line_total",
                 "from": item["line_total"], "to": val, "warnings": trace})
            item["line_total"] = val
        decisions[key] = "correct"
    if a.accept_total:
        decisions["total"] = "accept"
    if a.reject_total:
        decisions["total"] = "reject"
    if a.accept_all:
        for k in review.map_flags(data, result):
            decisions.setdefault(str(k), "accept")

    state = review.review_state(data, result, decisions)
    print(f"{data.get('receipt_id')} · {data.get('vendor')} · "
          f"TOTAL {_SYMBOL.get(data.get('currency') or 'MYR', data.get('currency'))} "
          f"{(data.get('receipt_total') or 0):.2f}")
    for row in state["rows"] + [state["total"]]:
        st = review.STATUS_LABEL[row["status"]]
        print(f"  {row['label']:<40} {st}")
        for f in row["flags"]:
            print(f"      - {f['type']}: {f['reason']}")

    rid = data.get("receipt_id") or "receipt"
    out_dir = Path(a.out) / f"{rid}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    try:
        csv_path, json_path, _ = export.export_receipt(
            data, result, decisions, out_dir)
    except export.ExportBlocked:
        print("\nBLOCKED — nothing written. Not accepted: "
              + review.blocker_text(state["blockers"]))
        return 2
    print(f"\nExported:\n  {csv_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
