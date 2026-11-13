"""Gradio demo. Run:  python -m src.ui.app        (opens http://127.0.0.1:7860)

UI features:
  - Image tab: upload receipt → Analyse (real PaddleOCR, local rules, no API)
  - JSON tab: upload pre-extracted JSON → Run detection (NO OCR — fixtures)
  - Line-item table: every row with its status and the full warning text
  - Review: pick ANY row or TOTAL → Accept / Reject / Correct; decisions
    can be changed. Rules in src/ui/review.py:
        no warning + no action → auto-accepted
        warning    + no action → auto-rejected
        Accept → accepted · Reject → rejected · Correct → corrected
  - Export: only when every row AND the TOTAL are accepted; one rejected
    entry blocks ledger.csv and audit_export.json. Each export goes to a
    new folder exports/<receipt>_<time>/ and the download links are
    cleared on every change, so an old file cannot be downloaded by
    mistake.
"""

import argparse
import contextvars
import html
import json
import os
import secrets
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from src.detect.detect import Detector
from src.detect.reference_builder import pool_stats
from src.extract.extract import to_json
from src.extract.hybrid_extract import extract_hybrid
from src.shared.models import Receipt, LineItem
from src.extract.name_normalizer import fallback_canonical
from src.shared import deepseek_client
from src.ui import export, highlight, review

_STATE = {
    "receipt": None,
    "result": None,
    "decisions": {},
    "anomaly_rows": [],
    "current_row": 0,
    "image": None,       # cleaned image path (image tab only) for evidence
}

# Per-browser-session state. Each Gradio session keeps its own dict in a
# gr.State; the handler wrappers in build_ui put it into this ContextVar
# for the duration of one call. Direct calls (tests, CLI) fall back to the
# module-level _STATE above. (Before 13/11 every visitor shared _STATE —
# fine for one person on localhost, wrong for a public site.)
_CUR = contextvars.ContextVar("docaudit_session", default=None)


def _S():
    s = _CUR.get()
    return _STATE if s is None else s


def _new_state():
    return {"receipt": None, "result": None, "decisions": {},
            "anomaly_rows": [], "current_row": 0, "image": None,
            "sid": secrets.token_hex(8)}


_ROOT = Path(__file__).resolve().parents[2]
# DOCAUDIT_PUBLIC=1 (hosted demo): no LLM button, exports go to a temp dir
# in a per-session folder, files are served only through the download box.
# DOCAUDIT_REQUIRE_LOGIN=1 (Hugging Face Space with hf_oauth): Analyse /
# Run detection need "Sign in with Hugging Face".
PUBLIC = os.environ.get("DOCAUDIT_PUBLIC") == "1"
REQUIRE_LOGIN = os.environ.get("DOCAUDIT_REQUIRE_LOGIN") == "1"
EXPORT_ROOT = Path(os.environ.get(
    "DOCAUDIT_EXPORT_DIR",
    str(Path(tempfile.gettempdir()) / "docaudit_exports") if PUBLIC else "exports"))


def _load_pool():
    """Reference price pool = SROIE pool + demo reference prices.

    demo/demo_price_pool.json holds ILLUSTRATIVE reference prices for the
    items used in demo/*.json. Without it no demo item has comparables,
    so the unusual-price check can never fire on the demo files.
    """
    pool = {}
    for rel in ("data/reference/pool.json", "demo/demo_price_pool.json"):
        ref_path = _ROOT / rel
        if ref_path.exists():
            try:
                pool.update(json.loads(ref_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    return pool


_item_key = review._item_key
_map_anomalies_to_rows = review.map_flags      # kept for older callers

_ICON = {"incorrect_total": "🔢", "duplicate": "🔁", "unusual_price": "💰",
         "needs_review": "⚠️", "total_unverified": "❔",
         "total_vs_duplicate": "🔢"}
_BADGE = {review.AUTO_ACCEPTED: "normal", review.AUTO_REJECTED: "pending",
          review.ACCEPTED: "accept", review.REJECTED: "reject",
          review.CORRECTED: "correct"}


def _warnings_html(flags):
    if not flags:
        return ""
    return "<br>".join(
        f"{_ICON.get(f['type'], '•')} <b>{html.escape(f['type'])}</b>: "
        f"{html.escape(f['reason'] or '')}" for f in flags)


def _badge(status):
    return (f'<span class="badge {_BADGE[status]}">'
            f'{review.STATUS_LABEL[status]}</span>')


def _build_html_table(data, result, selected_idx=None, decisions=None):
    """Every row with its effective status and ALL its warning reasons
    next to the flagged value (red cell)."""
    state = review.review_state(data, result, decisions or {})

    # Every row/cell sets BOTH background and text colour (dark theme).
    CELL_RED = "background-color:#EE3D5A;color:#ffffff;font-weight:bold;"
    ROW_ANOMALY_BG = "background-color:#fde2dd;color:#1b1b1b;"
    ROW_SELECTED_BG = "background-color:#fff3c4;color:#1b1b1b;"
    ROW_SELECTED_ANOMALY_BG = "background-color:#f9c6bd;color:#1b1b1b;"
    ROW_REJECTED_BG = "background-color:#e5e7e9;color:#1b1b1b;"

    rows_html = []
    for row, item in zip(state["rows"], data["items"]):
        i, flags = row["key"], row["flags"]
        cell = {f["cell"] for f in flags}
        name_style = CELL_RED if "name" in cell else ""
        unit_style = CELL_RED if "unit" in cell else ""
        line_style = CELL_RED if "line" in cell else ""

        if selected_idx == i:
            row_style = ROW_SELECTED_ANOMALY_BG if flags else ROW_SELECTED_BG
        elif row["status"] == review.REJECTED:
            row_style = ROW_REJECTED_BG
        elif flags:
            row_style = ROW_ANOMALY_BG
        else:
            row_style = ""
        if selected_idx == i:
            row_style += "outline:3px solid #0249E1;"

        icons = "".join(" " + _ICON.get(f["type"], "") for f in flags)
        rows_html.append(
            f'<tr style="{row_style}">'
            f'<td class="idx">{i}</td>'
            f'<td class="name" style="{name_style}">{html.escape(item["name"])}{icons}</td>'
            f'<td>{html.escape(item.get("canonical", "") or "")}</td>'
            f'<td>{item["qty"]}</td>'
            f'<td class="num" style="{unit_style}">{item["unit_price"]:.2f}</td>'
            f'<td class="num" style="{line_style}">{item["line_total"]:.2f}</td>'
            f'<td>{_badge(row["status"])}</td>'
            f'<td class="why">{_warnings_html(flags)}</td>'
            f'</tr>'
        )

    total = state["total"]
    t_flags = total["flags"]
    bg = "#ecf0f1"
    if total["status"] == review.REJECTED:
        bg = "#e5e7e9"
    elif t_flags:
        bg = "#fde2dd"
    if selected_idx == "total":
        bg = "#f9c6bd" if t_flags else "#fff3c4"
    total_row_style = ("font-weight:bold;border-top:2px solid #2c3e50;"
                       f"background-color:{bg};color:#1b1b1b;")
    if selected_idx == "total":
        total_row_style += "outline:3px solid #0249E1;"
    total_style = CELL_RED if t_flags else ""
    t_icons = "".join(" " + _ICON.get(f["type"], "") for f in t_flags)
    total_val = data.get("receipt_total") or 0

    rows_html.append(
        f'<tr style="{total_row_style}">'
        f'<td class="idx">T</td>'
        f'<td class="name">TOTAL{t_icons}</td>'
        f'<td></td><td></td><td></td>'
        f'<td class="num" style="{total_style}">{total_val:.2f}</td>'
        f'<td>{_badge(total["status"])}</td>'
        f'<td class="why">{_warnings_html(t_flags)}</td>'
        f'</tr>'
    )

    # Selectors are scoped under #da-wrap (an id) on purpose. Gradio 6
    # ships `.gradio-container-X .prose * {color: var(--body-text-color)}`,
    # which targets every <td> directly (specificity 0,2,0) and beat the
    # old `.da-table {color:…}` — in dark mode all cells were near-white
    # (#f4f4f5) on white / light-red rows. Measured in the browser 11/11.
    return f"""
    <style>
    #da-wrap .da-table {{ width:100%; border-collapse:collapse; font-size:13px; font-family:sans-serif;
                 background-color:#ffffff; color:#1b1b1b; }}
    #da-wrap .da-table th {{ background-color:#111111; color:#ffffff; padding:6px 8px; text-align:left;
                    border-bottom:2px solid #1a252f; }}
    #da-wrap .da-table td {{ padding:5px 8px; border-bottom:1px solid #ddd; color:#1b1b1b; vertical-align:top; }}
    #da-wrap .da-table td.idx {{ color:#555555; width:30px; text-align:center; }}
    #da-wrap .da-table .num {{ text-align:right; width:80px; }}
    #da-wrap .da-table .name {{ max-width:200px; }}
    #da-wrap .da-table td.why {{ font-size:12px; font-weight:normal; max-width:420px; }}
    #da-wrap .badge {{ padding:2px 6px; border-radius:3px; font-size:11px; font-weight:bold; white-space:nowrap; }}
    #da-wrap .badge.accept {{ background-color:#1e8449; color:#ffffff; }}
    #da-wrap .badge.reject {{ background-color:#c0392b; color:#ffffff; }}
    #da-wrap .badge.correct {{ background-color:#2471a3; color:#ffffff; }}
    #da-wrap .badge.pending {{ background-color:#a04000; color:#ffffff; }}
    #da-wrap .badge.normal {{ background-color:#bdc3c7; color:#2c3e50; }}
    </style>
    <div id="da-wrap">
    <table class="da-table">
        <thead>
            <tr>
                <th>#</th><th>Item</th><th>Canonical</th><th>Qty</th>
                <th>Unit (RM)</th><th>Line (RM)</th><th>Status</th><th>Why flagged</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    </div>
    """


_DECISION_LABEL = {"accept": "accepted", "reject": "rejected",
                   "correct": "corrected"}


def _get_anomaly_row_options(data, result, decisions):
    """Dropdown options: EVERY row and the TOTAL.

    Clean rows are listed too, so a reviewer can reject a row that was
    auto-accepted. Decided rows keep their decision in the label, so it
    can be changed.
    """
    state = review.review_state(data, result, decisions)
    options = []
    for row in state["rows"] + [state["total"]]:
        types = ", ".join(sorted({f["type"] for f in row["flags"]}))
        what = f"({types})" if types else "(no warning)"
        suffix = (f" — {_DECISION_LABEL[row['decision']]}"
                  if row["decision"] else "")
        head = "Row T: TOTAL" if row["key"] == "total" else row["label"]
        options.append(f"{head} {what}{suffix}")
    return options


def _pending_count(data, result, decisions):
    """Entries still blocking export (auto-rejected or rejected)."""
    return len(review.review_state(data, result, decisions)["blockers"])


def _parse_row_from_label(label):
    """'Row 3: ...' → 3 ; 'Row T: TOTAL ...' → 'total'."""
    if not label:
        return None
    head = label.split(":")[0].replace("Row", "").strip()
    if head == "T":
        return "total"
    try:
        return int(head)
    except ValueError:
        return None


def _label_for(options, key):
    """Dropdown label for row key (int or 'total'), or None."""
    for o in options:
        if _parse_row_from_label(o) == key:
            return o
    return None


def _gate_markdown(data, result, decisions):
    state = review.review_state(data, result, decisions)
    if state["can_export"]:
        n = len(state["rows"])
        return (f"✅ **Ready to export** — {n} row(s) and the TOTAL are "
                f"accepted (auto or by you).")
    lines = ["⛔ **Export blocked** — every row and the TOTAL must be "
             "accepted. Blocking now:"]
    for b in state["blockers"]:
        why = "; ".join(f"{f['type']}" for f in b["flags"]) or "rejected by you"
        lines.append(f"- **{html.escape(b['label'])}** — "
                     f"{review.STATUS_LABEL[b['status']]} ({html.escape(why)})")
    return "\n".join(lines)


def _row_details(key):
    """Everything the reviewer accepts or rejects for the selected entry."""
    data, result = _S()["receipt"], _S()["result"]
    if data is None or key is None:
        return ""
    state = review.review_state(data, result, _S()["decisions"])
    row = state["total"] if key == "total" else next(
        (r for r in state["rows"] if r["key"] == key), None)
    if row is None:
        return ""
    if key == "total":
        value = f"TOTAL RM {(data.get('receipt_total') or 0):.2f}"
        tc = (result or {}).get("total_check") or {}
        if tc.get("status") == "checked":
            value += " · total check ran"
    else:
        it = data["items"][key]
        value = (f"{html.escape(it['name'])} · {it['qty']} × "
                 f"RM {it['unit_price']:.2f} = RM {it['line_total']:.2f}")
    out = [f"**Selected: {html.escape(row['label'])}** — {value}",
           f"Status: **{review.STATUS_LABEL[row['status']]}**"]
    if row["flags"]:
        out.append(f"{len(row['flags'])} warning(s) — Accept means you "
                   f"confirm this entry despite ALL of them:")
        out += [f"- {_ICON.get(f['type'], '•')} **{f['type']}**: "
                f"{html.escape(f['reason'] or '')}" for f in row["flags"]]
    else:
        out.append("No warning. Reject it if the row is wrong anyway — "
                   "that blocks the export of the whole receipt.")
    return "\n\n".join(out)


def _review_outputs(log_msg=""):
    """(table, dropdown, decision row, corrected value, gate, details,
    export button, download, log) — the review panel, recomputed from
    the session state after every change. The download is ALWAYS cleared here: a
    file exported before a change must not be downloadable afterwards."""
    data, result = _S()["receipt"], _S()["result"]
    if data is None:
        return ("", gr.update(choices=[], value=None), gr.update(visible=False),
                "", "", "", gr.update(interactive=False), None, log_msg)
    decisions, cur = _S()["decisions"], _S()["current_row"]
    table_html = _build_html_table(data, result, selected_idx=cur,
                                   decisions=decisions)
    options = _get_anomaly_row_options(data, result, decisions)
    can = review.review_state(data, result, decisions)["can_export"]
    return (table_html,
            gr.update(choices=options, value=_label_for(options, cur)),
            gr.update(visible=True),
            "",
            _gate_markdown(data, result, decisions),
            _row_details(cur),
            gr.update(interactive=can),
            None,
            log_msg)


def _refresh(data, result, decisions, current_row):
    """Kept for older callers: review panel for the given state."""
    return _review_outputs()


def _annotated_image(selected=None):
    """Evidence image: every flagged box in red, the selected row in blue."""
    img_path = _S().get("image")
    data, result = _S()["receipt"], _S()["result"]
    if not img_path or data is None:
        return None
    img = highlight.annotate(img_path, [], _evidence_boxes(data, result))
    boxes = []
    if selected == "total":
        # the payable TOTAL row, plus any receipt-level incorrect_total
        # evidence (e.g. the altered 'Total Qty' subtotal row)
        boxes.append((data.get("_provenance") or {}).get("_total_region"))
        for a in result.get("anomalies", []):
            ev = a.get("evidence") or {}
            if a["anomaly_type"] == "incorrect_total" and "line_index" not in ev:
                boxes.append(ev.get("region"))
    elif isinstance(selected, int) and 0 <= selected < len(data["items"]):
        boxes.append(data["items"][selected].get("region"))
    boxes = [b for b in boxes if b and highlight.parse_box(b)]
    if boxes:
        img = highlight.annotate_on(img, boxes, colour=(30, 90, 220), width=5)
    return img


def select_row(row_label):
    """Dropdown pick → highlight that row in the table and on the image."""
    idx = _parse_row_from_label(row_label)
    if _S()["receipt"] is None or idx is None:
        return gr.update(), gr.update()
    _S()["current_row"] = idx
    table = _build_html_table(_S()["receipt"], _S()["result"],
                              selected_idx=idx, decisions=_S()["decisions"])
    img = _annotated_image(idx)
    return table, (img if img is not None else gr.update())


def show_details(row_label):
    return _row_details(_parse_row_from_label(row_label))


def _anomaly_markdown(result):
    lines = [f"**{a['anomaly_type']}** ({a['confidence']:.2f}) — {a['reason']}"
             for a in result["anomalies"]]
    tc = result.get("total_check") or {}
    if tc.get("status") in review.UNVERIFIED_TOTAL:
        lines.append(f"❔ **total not verified** — {tc.get('reason', '')}")
    return "\n\n".join(lines) or "No anomalies detected."


def _load(data, result, image=None):
    """New receipt: forget every decision, selection and export link."""
    _S()["receipt"] = data
    _S()["result"] = result
    _S()["decisions"] = {}
    _S()["current_row"] = 0 if data.get("items") else "total"
    _S()["image"] = image
    flag_map = review.map_flags(data, result)
    _S()["anomaly_rows"] = sorted(k for k in flag_map if k != "total")


def _empty(msg):
    _S().update({"receipt": None, "result": None, "decisions": {},
                   "image": None, "anomaly_rows": [], "current_row": 0})
    return (msg, None, "") + _review_outputs()


def process(image_path):
    if not image_path:
        return _empty("Upload a receipt image first.")

    try:
        receipt = extract_hybrid(image_path, skip_api=True)
    except Exception as e:
        return _empty(f"Extraction failed: {e}")

    pool = _load_pool()
    result = Detector(price_pool=pool).run(receipt)
    data = to_json(receipt)

    ann_image = receipt.provenance.get("_clean_image", image_path)
    _load(data, result, ann_image if receipt.regions else None)
    ann = _annotated_image(None)

    stats = pool_stats(pool)
    mapping = receipt.provenance.get("_mapping", "?")
    unparsed = receipt.provenance.get("_unparsed_rows", [])
    if not data["items"]:
        mapping += " · ⚠️ no line items recognised — check the image"
    if unparsed:
        mapping += (f" · ⚠️ {len(unparsed)} item row(s) could not be read: "
                    + "; ".join(u["raw_text"] for u in unparsed[:3]))
    header = (f"**{data['vendor'] or 'Unknown vendor'}** · "
              f"{data['date'] or 'no date'} · "
              f"Total RM {data['receipt_total'] or 0:.2f}\n\n"
              f"Source: image → PaddleOCR · Mapping: {mapping}\n\n"
              f"{result['summary']}\n\n"
              f"Price pool: {stats['items']} items "
              f"({stats['observations']} observations)")

    return (header, ann, _anomaly_markdown(result)) + _review_outputs(
        f"Loaded {data.get('receipt_id')} — previous decisions cleared.")


def _evidence_boxes(data, result):
    """Bbox strings to outline on the image: every flagged row + totals."""
    boxes = []
    flag_map = review.map_flags(data, result)
    for idx in flag_map:
        if idx == "total":
            continue
        reg = data["items"][idx].get("region")
        if reg:
            boxes.append(reg)
    for a in result.get("anomalies", []):
        reg = (a.get("evidence") or {}).get("region", "")
        if reg and highlight.parse_box(reg):
            boxes.append(reg)
    return boxes


def _normalise_json(data, json_path):
    """Hand-written / injected JSON files lack fields the UI relies on."""
    data.setdefault("receipt_id", Path(json_path).stem)
    data.setdefault("currency", "MYR")
    for item in data.get("items", []):
        item.setdefault("canonical", fallback_canonical(item.get("name", "")))
        item.setdefault("review", "unreviewed")
        item.setdefault("reason", "")
        item.setdefault("region", "")
    return data


def process_json(json_path):
    """JSON fixture flow: NO OCR — the file already holds the fields."""
    if not json_path:
        return _empty("Upload a JSON file first.")

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as e:
        return _empty(f"JSON load failed: {e}")

    if not isinstance(data, dict) or "items" not in data:
        return _empty("This JSON is not a receipt (no 'items' list).")
    data = _normalise_json(data, json_path)
    receipt = _receipt_from_json(data)
    pool = _load_pool()
    result = Detector(price_pool=pool).run(receipt)
    _load(data, result, None)

    stats = pool_stats(pool)
    header = (f"**{data.get('vendor') or 'Unknown vendor'}** · "
              f"{data.get('date') or 'no date'} · "
              f"Total RM {data.get('receipt_total') or 0:.2f}\n\n"
              f"Source: JSON fixture (no OCR)\n\n"
              f"{result['summary']}\n\n"
              f"Price pool: {stats['items']} items "
              f"({stats['observations']} observations)")

    # JSON has no image: clear the previous receipt's evidence picture
    return (header, None, _anomaly_markdown(result)) + _review_outputs(
        f"Loaded {data.get('receipt_id')} — previous decisions cleared.")


def _receipt_from_json(data):
    items = []
    for i in data.get("items", []):
        items.append(LineItem(
            name=i["name"], canonical=i.get("canonical", ""),
            qty=i["qty"], unit_price=i["unit_price"],
            line_total=i["line_total"], raw_text=i.get("raw", ""),
            region=i.get("region", ""),
            ocr_confidence=i.get("ocr_confidence", 1.0),
            mismatch=i.get("mismatch", False),
        ))
    return Receipt(
        vendor=data.get("vendor"), date=data.get("date"),
        currency=data.get("currency", "MYR"),
        items=items, receipt_total=data.get("receipt_total"),
        tax=data.get("tax"), printed_subtotal=data.get("printed_subtotal"),
        receipt_id=data.get("receipt_id", ""),
        image_path=data.get("_provenance", {}).get("_clean_image", ""),
        regions=[], provenance=dict(data.get("_provenance", {})),
    )


def _after_decision(msg):
    return _review_outputs(msg)


def _decide(row_label, decision):
    idx = _parse_row_from_label(row_label)
    if idx is None or _S()["receipt"] is None:
        return None
    _S()["decisions"][str(idx)] = decision
    _S()["current_row"] = idx          # 'total' included: TOTAL row highlights
    return idx


def _row_name(idx):
    return "TOTAL" if idx == "total" else f"Row {idx}"


def _left_msg():
    n = _pending_count(_S()["receipt"], _S()["result"], _S()["decisions"])
    return "ready to export" if n == 0 else f"{n} entr{'y' if n == 1 else 'ies'} still blocking export"


def decide_accept(row_label):
    idx = _decide(row_label, "accept")
    if idx is None:
        return _after_decision("Select a row first.")
    return _after_decision(f"{_row_name(idx)}: Accepted · {_left_msg()}")


def decide_reject(row_label):
    idx = _decide(row_label, "reject")
    if idx is None:
        return _after_decision("Select a row first.")
    return _after_decision(f"{_row_name(idx)}: Rejected · {_left_msg()}")


def decide_correct(row_label, correct_value):
    """Correct = Accept with a new value. The observed value and the
    warnings are kept in `_corrections`; the checks are not re-run."""
    idx = _parse_row_from_label(row_label)
    if idx is None or _S()["receipt"] is None:
        return _after_decision("Select a row first.")
    try:
        val = round(float(correct_value), 2)
    except (ValueError, TypeError):
        return _after_decision("Enter a number in 'Corrected value' first.")
    data = _S()["receipt"]
    flags = review.map_flags(data, _S()["result"]).get(idx, [])
    trace = [f"{f['type']}: {f['reason']}" for f in flags]
    if idx == "total":
        data.setdefault("_corrections", []).append(
            {"field": "receipt_total", "from": data.get("receipt_total"),
             "to": val, "warnings": trace})
        data["receipt_total"] = val
    else:
        item = data["items"][idx]
        data.setdefault("_corrections", []).append(
            {"row": idx, "field": "line_total", "from": item["line_total"],
             "to": val, "warnings": trace})
        item["line_total"] = val
    _decide(row_label, "correct")
    return _after_decision(f"{_row_name(idx)}: Corrected to {val:.2f} · {_left_msg()}")


def accept_all_normal():
    """Accept every flagged entry (rows and TOTAL) that has no decision
    yet. Explicit Rejects are NOT overridden — they keep blocking."""
    if _S()["receipt"] is None:
        return _after_decision("Nothing loaded.")
    flag_map = review.map_flags(_S()["receipt"], _S()["result"])
    n = 0
    for idx in flag_map:
        if str(idx) not in _S()["decisions"]:
            _S()["decisions"][str(idx)] = "accept"
            n += 1
    return _after_decision(f"Accepted {n} remaining flagged entr"
                           f"{'y' if n == 1 else 'ies'} · {_left_msg()}")


def navigate_row(direction):
    """Move current row selection up or down."""
    if _S()["receipt"] is None or not _S()["receipt"]["items"]:
        return _review_outputs()
    n_items = len(_S()["receipt"]["items"])
    cur = _S()["current_row"]
    cur = n_items - 1 if cur == "total" else cur
    _S()["current_row"] = max(0, min(cur + direction, n_items - 1))
    return _review_outputs()


def navigate_anomaly(direction):
    """Jump to next or previous anomaly row."""
    if _S()["receipt"] is None or not _S()["anomaly_rows"]:
        return _review_outputs()
    current = _S()["current_row"]
    if current == "total":
        current = len(_S()["receipt"]["items"])
    rows = _S()["anomaly_rows"]
    if direction > 0:
        nxt = [r for r in rows if r > current]
        new_row = nxt[0] if nxt else current
    else:
        prev = [r for r in rows if r < current]
        new_row = prev[-1] if prev else current
    _S()["current_row"] = new_row
    return _review_outputs()


def _apply_decisions(data, result, decisions):
    """Receipt stamped with the effective status of every row + TOTAL."""
    return review.apply_review(data, result, decisions)


def _prune_exports(max_age=3600):
    """Public mode: delete export folders older than `max_age` seconds."""
    import shutil
    import time
    if not EXPORT_ROOT.exists():
        return
    cutoff = time.time() - max_age
    for d in EXPORT_ROOT.glob("*/*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def do_export():
    """(message, download files). Re-checks the gate at click time."""
    if _S()["receipt"] is None:
        return "Nothing to export.", None
    data, result = _S()["receipt"], _S()["result"]
    rid = data.get("receipt_id") or "receipt"
    if PUBLIC:
        _prune_exports()
    sid = _S().get("sid")
    base = EXPORT_ROOT / sid if sid else EXPORT_ROOT
    out_dir = base / f"{rid}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    try:
        csv_path, json_path, stamped = export.export_receipt(
            data, result, _S()["decisions"], out_dir)
    except export.ExportBlocked as e:
        state = review.review_state(data, result, _S()["decisions"])
        return ("⛔ Nothing written. Export blocked by: "
                + review.blocker_text(state["blockers"])), None
    s = export.export_summary([stamped], [result])
    where = ("Files: ledger.csv, audit_export.json (download below)"
             if PUBLIC else f"CSV: {csv_path}\nAudit: {json_path}")
    return (f"✅ Exported {rid}\n{where}\n\n"
            f"Rows {s['total_items']} · auto-accepted {s['auto_accepted']} · "
            f"accepted {s['accepted']} · corrected {s['corrected']}\n"
            f"TOTAL: {stamped['total_review']}"), [csv_path, json_path]


def health():
    h = deepseek_client.health_check()
    return "LLM OK" if h["ok"] else f"LLM FAILED: {h['error']}"


def _in_session(fn):
    """Gradio handler running `fn` against THIS browser session's state.

    Signature (st, *args) → (*fn outputs, st): the gr.State is the first
    input and the last output, so every visitor has their own receipt,
    decisions and export folder.
    """
    def handler(st, *args):
        st = st if isinstance(st, dict) else _new_state()
        token = _CUR.set(st)
        try:
            out = fn(*args)
        finally:
            _CUR.reset(token)
        out = out if isinstance(out, tuple) else (out,)
        return (*out, st)
    handler.__name__ = fn.__name__
    return handler


_LOGIN_MSG = ("🔒 Please **Sign in with Hugging Face** (button at the top) "
              "to analyse receipts.")


def _login_gate(fn):
    """Wraps process / process_json: on the hosted Space they run only for
    a signed-in visitor. gr.OAuthProfile is injected by Gradio when the
    Space has `hf_oauth: true`; None means not signed in."""
    def gated(st, path, profile: gr.OAuthProfile | None = None):
        if REQUIRE_LOGIN and profile is None:
            st = st if isinstance(st, dict) else _new_state()
            token = _CUR.set(st)
            try:
                out = _empty(_LOGIN_MSG)
            finally:
                _CUR.reset(token)
            return (*out, st)
        return _in_session(fn)(st, path)
    gated.__name__ = fn.__name__
    return gated


_PUBLIC_NOTE = (
    "**Public demo.** Receipts you upload are processed on this server only "
    "to show the result; uploads and exports are deleted automatically "
    "within about an hour — do not upload "
    "receipts with card numbers or other personal data. Warnings are signals "
    "to review, not proof of fraud.")


# Brand palette (same as site/index.html and the sign-in page):
# 0249E1 blue · 80B0EC sky · FFFFFF · 111111 ink · DAFB71 lime · EE3D5A red.
# Every step sits on a white "ticket" card (notched dotted cuts) on the
# blue page, like /login and /signup. Card colours are set explicitly so
# Gradio's dark mode cannot turn them dark.
_BRAND_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;600;700;800&display=swap');
body, .gradio-container { background:#0249E1 !important; }
.gradio-container { font-family:"Urbanist",-apple-system,"Segoe UI","PingFang SC",sans-serif !important;
  max-width:1240px !important; margin:0 auto !important; padding:20px 16px 48px !important; }
.gradio-container footer { color:#dfe8ff !important; }
.da-band { border-radius:28px; padding:22px 26px; color:#fff; margin-bottom:6px;
  background: radial-gradient(circle at 8% 140%, #EE3D5A 0 18%, transparent 19%),
              radial-gradient(circle at 96% -60%, #DAFB71 0 24%, transparent 25%),
              linear-gradient(135deg,#0a3fc0,#0249E1 60%,#1f63f5); }
.da-band small { font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:12px; opacity:.9 }
.da-band h1 { margin:4px 0 6px; font-size:34px; letter-spacing:-.02em; color:#fff }
.da-band p { margin:0; color:#e6eeff; font-size:15px }
.da-band a { color:#DAFB71; font-weight:700 }
.da-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px }
.da-chips span { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  border-radius:999px; padding:3px 10px; font-size:13px; color:#fff }
.da-card { background:#fff !important; border-radius:28px !important; padding:16px 18px 20px !important;
  box-shadow:0 24px 48px rgba(0,20,80,.25); border:0 !important;
  --block-background-fill:#ffffff; --background-fill-primary:#ffffff;
  --background-fill-secondary:#f5f7fb; --block-border-color:#e3e6ee;
  --border-color-primary:#e3e6ee; --body-text-color:#111111;
  --body-text-color-subdued:#5b6170; --block-label-text-color:#5b6170;
  --block-title-text-color:#111111; --input-background-fill:#f5f7fb;
  --color-accent:#0249E1; --color-accent-soft:#e6eeff; --link-text-color:#0249E1;
  --table-even-background-fill:#fff; --table-odd-background-fill:#f8f9fc;
  --block-label-background-fill:#e6eeff; --block-label-text-color:#0249E1;
  --block-label-border-color:#e6eeff;
  color:#111111; }
.da-card .prose, .da-card .prose * { color:#111111; }
.da-card .prose a { color:#0249E1 !important; font-weight:600 }
.da-step { display:flex; align-items:center; gap:10px; font-weight:800; font-size:19px;
  color:#111; margin:2px 2px 4px }
.da-step b { display:inline-grid; place-items:center; width:30px; height:30px; border-radius:50%;
  background:#0249E1; color:#fff; font-size:15px }
.da-step.lime b { background:#DAFB71; color:#111 }
.da-step span { font-weight:500; font-size:14px; color:#5b6170 }
.da-cut { position:relative; height:24px; margin:4px -18px; overflow:hidden }
.da-cut:after { content:""; position:absolute; left:24px; right:24px; top:11px; border-top:2px dotted #80B0EC }
.da-cut i { position:absolute; top:0; width:24px; height:24px; border-radius:50%; background:#0249E1 }
.da-cut i:first-child { left:-12px } .da-cut i:last-child { right:-12px }
.da-card button { border-radius:999px !important; font-weight:700 !important; }
#da-analyse, #da-run, #da-export { background:#0249E1 !important; color:#fff !important; border:0 !important; }
#da-accept { background:#DAFB71 !important; color:#111111 !important; border:0 !important; }
#da-reject { background:#EE3D5A !important; color:#fff !important; border:0 !important; }
#da-export:disabled { background:#80B0EC !important; color:#111111 !important; opacity:.75; }
#da-gate .prose p:first-child { font-size:16px }
.da-row { gap:16px !important; align-items:stretch }
@media (max-width: 900px) { .da-band h1 { font-size:26px } }
"""

_CUT = '<div class="da-cut"><i></i><i></i></div>'


def _step(n, title, hint="", lime=False):
    cls = "da-step lime" if lime else "da-step"
    return gr.HTML(f'<div class="{cls}"><b>{n}</b>{title}'
                   f'{f"<span>· {hint}</span>" if hint else ""}</div>')


def _band_html():
    links = ""
    if os.environ.get("DOCAUDIT_ACCOUNTS") == "1":
        links = ' · <a href="/">About DocAudit</a> · <a href="/logout">Sign out</a>'
    note = (f"<p style='margin-top:8px;font-size:13px'>{html.escape(_PUBLIC_NOTE).replace('**', '')}</p>"
            if PUBLIC else "")
    return f"""<div class="da-band">
  <small>DocAudit · receipt check{links}</small>
  <h1>Check a receipt, then decide what to export</h1>
  <p>Read with PaddleOCR · checked by rules · every warning shows its calculation and the
     spot on the image. No warning → auto-accepted · warning → auto-rejected until you
     Accept · export only when every row <b>and</b> the TOTAL are accepted.</p>
  <div class="da-chips"><span>🔢 incorrect total</span><span>🔁 duplicate</span>
    <span>💰 unusual price</span><span>⚠️ low OCR confidence</span><span>❔ total not verified</span></div>
  {note}
</div>"""


def build_ui():
    # public mode: Gradio deletes uploaded/served files older than 1 h,
    # checked every 30 min
    blocks_kw = {"delete_cache": (1800, 3600)} if PUBLIC else {}
    with gr.Blocks(title="DocAudit — receipt check", **blocks_kw) as demo:
        gr.HTML(_band_html())
        if REQUIRE_LOGIN:
            gr.LoginButton()
        session = gr.State(None)

        with gr.Row(elem_classes="da-row"):
            with gr.Column(scale=4, elem_classes="da-card"):
                _step(1, "Upload", "photo, scan or JSON")
                with gr.Tabs():
                    with gr.Tab("Receipt photo"):
                        image_in = gr.Image(type="filepath", label="Receipt photo",
                                            height=360)
                        run_btn = gr.Button("Analyse", variant="primary",
                                            elem_id="da-analyse")
                        health_btn = gr.Button("LLM health check (optional)",
                                               visible=not PUBLIC)
                        health_out = gr.Textbox(label="API status", lines=2,
                                                visible=not PUBLIC)
                    with gr.Tab("JSON fixture (no OCR)"):
                        json_in = gr.File(label="Receipt JSON", file_types=[".json"])
                        json_btn = gr.Button("Run detection", variant="primary",
                                             elem_id="da-run")
                        gr.Markdown("Pre-extracted / error-injected receipts "
                                    "(`demo/*.json`). This tab skips OCR.")

            with gr.Column(scale=7, elem_classes="da-card"):
                _step(2, "What DocAudit read", "red = flagged · blue = selected")
                header_out = gr.Markdown()
                img_out = gr.Image(label="Evidence regions", type="pil", height=380)
                gr.HTML(_CUT)
                anomalies_out = gr.Markdown()

        with gr.Column(elem_classes="da-card"):
            _step(3, "Review every row and the TOTAL", "pick a row → Accept / Reject")
            table_html = gr.HTML()
            gr.HTML(_CUT)
            with gr.Row(visible=False) as decision_row:
                row_selector = gr.Dropdown(
                    label="Row or TOTAL to review (all rows listed)",
                    choices=[], scale=3)
                with gr.Column(scale=2, min_width=160):
                    accept_btn = gr.Button("Accept", variant="primary",
                                           elem_id="da-accept")
                    reject_btn = gr.Button("Reject", variant="stop",
                                           elem_id="da-reject")
                with gr.Column(scale=2, min_width=160):
                    correct_input = gr.Textbox(
                        label="Corrected value (line total / TOTAL)")
                    correct_btn = gr.Button("Correct")
            row_details = gr.Markdown()
            accept_all_btn = gr.Button("Accept all remaining flagged entries")

        with gr.Column(elem_classes="da-card"):
            _step(4, "Export", "only when nothing is blocking", lime=True)
            gate_md = gr.Markdown(elem_id="da-gate")
            export_btn = gr.Button("Export receipt (CSV + JSON)",
                                   variant="primary", interactive=False,
                                   elem_id="da-export")
            with gr.Row():
                export_out = gr.Textbox(label="Export result", lines=4, scale=3)
                download = gr.File(label="Download (cleared after any change)",
                                   file_count="multiple", interactive=False, scale=2)
            decision_log = gr.Textbox(label="Decision log", lines=1)

        review_outputs = [table_html, row_selector, decision_row,
                          correct_input, gate_md, row_details, export_btn,
                          download, decision_log]

        # every handler gets this session's state first and returns it
        # last (see _in_session); a new receipt also clears the old export
        clear_export = lambda: ""  # noqa: E731
        # Only the Space (REQUIRE_LOGIN) gets the OAuth-annotated gate:
        # Gradio switches on OAuth — and needs gradio[oauth] — as soon as
        # any handler takes a gr.OAuthProfile, so local runs use the plain one.
        load = _login_gate if REQUIRE_LOGIN else _in_session
        run_btn.click(
            load(process), inputs=[session, image_in],
            outputs=[header_out, img_out, anomalies_out] + review_outputs + [session]
        ).then(clear_export, outputs=export_out)
        json_btn.click(
            load(process_json), inputs=[session, json_in],
            outputs=[header_out, img_out, anomalies_out] + review_outputs + [session]
        ).then(clear_export, outputs=export_out)
        if not PUBLIC:
            health_btn.click(health, outputs=health_out)

        for btn, fn, inputs in (
                (accept_btn, decide_accept, [row_selector]),
                (reject_btn, decide_reject, [row_selector]),
                (correct_btn, decide_correct, [row_selector, correct_input]),
                (accept_all_btn, accept_all_normal, [])):
            btn.click(_in_session(fn), inputs=[session] + inputs,
                      outputs=review_outputs + [session]).then(
                clear_export, outputs=export_out)

        # user picks a row → highlight it in the table and on the image
        row_selector.input(_in_session(select_row),
                           inputs=[session, row_selector],
                           outputs=[table_html, img_out, session])
        row_selector.input(_in_session(show_details),
                           inputs=[session, row_selector],
                           outputs=[row_details, session])

        export_btn.click(_in_session(do_export), inputs=[session],
                         outputs=[export_out, download, session])

    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description="DocAudit web UI")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("DOCAUDIT_PORT", 7860)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the browser automatically")
    args = ap.parse_args(argv)
    try:                      # show Gradio's 'Running on…' line at once,
        sys.stdout.reconfigure(line_buffering=True)   # even when piped
    except (AttributeError, ValueError):
        pass
    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 60)
    print(f" DocAudit UI starting → open {url}")
    print(" Ready when the line 'Running on local URL' appears below.")
    print(" Stop with Ctrl+C.")
    print("=" * 60 + "\n", flush=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    build_ui().launch(server_name=args.host, server_port=args.port,
                      inbrowser=not args.no_browser, css=_BRAND_CSS,
                      allowed_paths=[str(EXPORT_ROOT.resolve())])


if __name__ == "__main__":
    main()
