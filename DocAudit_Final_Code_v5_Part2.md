# DocAudit — Final Pipeline & Code (v5, Part 2 of 2)

**UI · Evaluation · Tests · What Else You Need · Build Order**

> **Part 1** contains Parts 1–3A: the pipeline, project structure, config, shared modules, and all extraction + detection code.

---

## PART 3 — FILE CONTENTS (B: UI, EVAL, TESTS)

### 3.23 `src/ui/highlight.py`

```python
"""Draws PaddleOCR boxes onto the receipt image.

This is what makes evidence visible: a flagged field points at the
exact region it came from. Because the boxes are PADDLEOCR'S OWN
detections — not model guesses — the highlight is provably correct.
"""

from PIL import Image, ImageDraw


def annotate(image_path: str, regions: list, flagged_boxes: list,
             colour=(220, 40, 40), width: int = 3):
    """Returns an annotated copy. flagged_boxes is a list of bbox strings."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    flagged = set(flagged_boxes)

    for r in regions:
        bbox = r.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        key = ",".join(str(int(v)) for v in bbox)
        if key in flagged:
            draw.rectangle(bbox, outline=colour, width=width)
    return img


def parse_box(box_str: str):
    """'10,20,110,60' → [10, 20, 110, 60]"""
    if not box_str:
        return None
    try:
        parts = [int(float(v)) for v in box_str.split(",")]
        return parts if len(parts) == 4 else None
    except (ValueError, AttributeError):
        return None
```

### 3.24 `src/ui/export.py` ★ NEW

```python
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
```

### 3.25 `src/ui/app.py` ★ NEW

```python
"""Gradio demo. Run:  python -m src.ui.app

THREE DISPOSITIONS, not two — the third is the point.

  Accept   → row goes to the ledger as-is
  Correct  → human fixes the value; the fix becomes authoritative
  Reject   → flag is wrong; row is dropped from the ledger

A binary accept/reject gives the same outcome for "the system found a
real error" and "the system invented one" — which loses the data you
need to separate them. The third option is what lets you report
correction vs rejection rates.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from src.detect.detect import Detector
from src.detect.reference_builder import build_pool, pool_stats
from src.extract.extract import extract_one, to_json
from src.shared import deepseek_client
from src.ui import export, highlight

_STATE = {"receipt": None, "result": None, "decisions": {}}


def _load_pool():
    """Reference pool for Check 3. Empty pool = Check 3 flags nothing."""
    ref_path = Path("data/reference/pool.json")
    if ref_path.exists():
        try:
            return json.loads(ref_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def process(image_path: str):
    if not image_path:
        return "Upload a receipt image first.", None, "", None

    try:
        receipt = extract_one(image_path)
    except Exception as e:
        return f"Extraction failed: {e}", None, "", None

    pool = _load_pool()
    result = Detector(price_pool=pool).run(receipt)

    data = to_json(receipt)
    _STATE["receipt"] = data
    _STATE["result"] = result
    _STATE["decisions"] = {}

    flagged = [i["region"] for i in data["items"] if i["reason"]]
    ann = highlight.annotate(image_path, receipt.regions, flagged) \
        if receipt.regions else None

    table = [[i["name"], i["canonical"], i["qty"], i["unit_price"],
              i["line_total"], i["review"], i["reason"]]
             for i in data["items"]]

    stats = pool_stats(pool)
    header = (f"**{data['vendor'] or 'Unknown vendor'}** · "
              f"{data['date'] or 'no date'} · "
              f"Total RM {data['receipt_total'] or 0:.2f}\n\n"
              f"{result['summary']}\n\n"
              f"Price pool: {stats['items']} items "
              f"({stats['observations']} observations)")

    anomalies = "\n".join(
        f"**{a['anomaly_type']}** ({a['confidence']}) — {a['reason']}"
        for a in result["anomalies"]) or "No anomalies detected."

    return header, ann, anomalies, table


def decide(decision: str, row_index: int):
    """Record the human disposition for one row."""
    if _STATE["receipt"] is None:
        return "Nothing loaded."
    items = _STATE["receipt"]["items"]
    idx = int(row_index)
    if not 0 <= idx < len(items):
        return f"Row {idx} out of range (0–{len(items) - 1})."

    _STATE["decisions"][idx] = decision
    item = items[idx]
    item["review"] = {"accept": "accepted",
                      "correct": "corrected",
                      "reject": "rejected"}[decision]
    return f"Row {idx} '{item['name']}' → {item['review']}"


def do_export():
    if _STATE["receipt"] is None:
        return "Nothing to export."
    receipts = [_STATE["receipt"]]
    results = [_STATE["result"]]
    csv_path = export.to_ledger_csv(receipts, "ledger.csv")
    json_path = export.to_audit_json(receipts, results, "audit_export.json")
    s = export.export_summary(receipts, results)
    return (f"Exported.\nCSV: {csv_path}\nAudit: {json_path}\n\n"
            f"Items {s['total_items']} · flagged {s['flagged']} · "
            f"confirmed {s['confirmed']} · rejected {s['rejected']} · "
            f"review load {s['review_load']}")


def health():
    h = deepseek_client.health_check()
    return "DeepSeek OK" if h["ok"] else f"DeepSeek FAILED: {h['error']}"


def build_ui():
    with gr.Blocks(title="DocAudit — MYR Receipt Anomaly Detection") as demo:
        gr.Markdown("# DocAudit — MYR Receipt Anomaly Detection\n"
                    "Read with PaddleOCR · Structure with DeepSeek · "
                    "Decide with code · Explain with templates")

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Receipt photo")
                run_btn = gr.Button("Analyse", variant="primary")
                health_btn = gr.Button("DeepSeek health check")
                health_out = gr.Textbox(label="API status", lines=2)

            with gr.Column(scale=2):
                header_out = gr.Markdown()
                img_out = gr.Image(label="Evidence regions", type="pil")
                anomalies_out = gr.Markdown(label="Anomalies")

        table_out = gr.Dataframe(
            headers=["Item", "Canonical", "Qty", "Unit (RM)", "Line (RM)",
                     "Review", "Reason"],
            label="Line items", wrap=True)

        with gr.Row():
            row_idx = gr.Number(label="Row index", value=0, precision=0)
            accept_btn = gr.Button("Accept")
            correct_btn = gr.Button("Correct")
            reject_btn = gr.Button("Reject")
            decision_out = gr.Textbox(label="Decision log")

        export_btn = gr.Button("Export confirmed rows", variant="primary")
        export_out = gr.Textbox(label="Export result", lines=4)

        run_btn.click(process, inputs=image_in,
                      outputs=[header_out, img_out, anomalies_out, table_out])
        health_btn.click(health, outputs=health_out)
        accept_btn.click(lambda i: decide("accept", i), inputs=row_idx,
                         outputs=decision_out)
        correct_btn.click(lambda i: decide("correct", i), inputs=row_idx,
                          outputs=decision_out)
        reject_btn.click(lambda i: decide("reject", i), inputs=row_idx,
                         outputs=decision_out)
        export_btn.click(do_export, outputs=export_out)

    return demo


if __name__ == "__main__":
    build_ui().launch()
```

### 3.26 `eval/error_injector.py`

```python
"""Generates the labelled error set.

No public dataset carries these three anomaly types as ground truth,
so this self-built set is what actually evaluates the system.
Every error receipt is LABELLED.
"""

import copy
import json
import random
from pathlib import Path

random.seed(42)


def inject_wrong_total(receipt, magnitude: float = 2.0):
    r = copy.deepcopy(receipt)
    delta = round(random.choice([-1, 1]) * magnitude, 2)
    r["receipt_total"] = round(r["receipt_total"] + delta, 2)
    return r, {"anomaly_type": "incorrect_total", "expected_delta": delta}


def inject_duplicate_item(receipt):
    r = copy.deepcopy(receipt)
    idx = random.randrange(len(r["items"]))
    dup = copy.deepcopy(r["items"][idx])
    r["items"].insert(idx + 1, dup)
    r["receipt_total"] = round(r["receipt_total"] + dup["line_total"], 2)
    return r, {"anomaly_type": "duplicate", "duplicated_index": idx}


def inject_inflated_price(receipt, factor: float = 2.5):
    r = copy.deepcopy(receipt)
    idx = random.randrange(len(r["items"]))
    item = r["items"][idx]
    old_line = item["line_total"]
    item["unit_price"] = round(item["unit_price"] * factor, 2)
    item["line_total"] = round(item["qty"] * item["unit_price"], 2)
    r["receipt_total"] = round(
        r["receipt_total"] - old_line + item["line_total"], 2)
    return r, {"anomaly_type": "unusual_price",
               "inflated_index": idx, "factor": factor}


INJECTORS = {
    "incorrect_total": inject_wrong_total,
    "duplicate": inject_duplicate_item,
    "unusual_price": inject_inflated_price,
}


def build_error_set(normal_paths: list, out_dir: str) -> list:
    """Three faults per normal receipt → ~150 labels from 50 receipts."""
    out = Path(out_dir)
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    labels = []

    for i, path in enumerate(normal_paths):
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
        for kind, fn in INJECTORS.items():
            mutated, meta = fn(receipt)
            rid = f"err_{i:03d}_{kind}"
            (out / "receipts" / f"{rid}.json").write_text(
                json.dumps(mutated, indent=2, ensure_ascii=False),
                encoding="utf-8")
            labels.append({"receipt_id": rid, "anomaly_type": kind,
                           "label": 1, **meta})

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8")
    return labels
```

### 3.27 `eval/compute_metrics.py`

```python
"""The four metrics.

    Precision = TP/(TP+FP)      Recall = TP/(TP+FN)
    FPR       = FP/(FP+TN)      Review load = Alerts/Total receipts

Run AFTER extraction is cached — that is what makes the numbers stable.
"""

import json
import pathlib


def confusion(preds: list, labels: list, atype: str) -> dict:
    """Counts for ONE anomaly type.

    Labelled positives the system never flagged are counted as false
    negatives — otherwise recall is inflated by omission.
    """
    truth = {(l["receipt_id"], l["anomaly_type"]): l["label"] for l in labels}
    tp = fp = fn = tn = 0
    seen = set()

    for p in preds:
        if p["anomaly_type"] != atype:
            continue
        key = (p["receipt_id"], atype)
        seen.add(key)
        actual = truth.get(key, 0)
        if p["label"] == 1 and actual == 1:
            tp += 1
        elif p["label"] == 1 and actual == 0:
            fp += 1
        elif p["label"] == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    for key, actual in truth.items():
        if actual == 1 and key[1] == atype and key not in seen:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(preds: list, labels: list, total_receipts: int) -> dict:
    out = {}
    for atype in ("incorrect_total", "duplicate", "unusual_price"):
        c = confusion(preds, labels, atype)
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[atype] = {"precision": round(prec, 4), "recall": round(rec, 4),
                      "fpr": round(fpr, 4), "f1": round(f1, 4), **c}

    alerts = sum(1 for p in preds if p["label"] == 1)
    out["review_load"] = round(alerts / max(total_receipts, 1), 4)
    return out


def write_report(m: dict, path: str = "eval/results/metrics.json"):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return str(p)
```

### 3.28 `tests/test_name_normalizer.py` ★ NEW

```python
"""Locks the fallback normaliser and the alignment logic.

The DeepSeek path is not tested here — it is model output, and it is
validated by comparing its results against these fallback rules.
"""

from src.extract.name_normalizer import (fallback_canonical, _align,
                                         canonical_key)
from src.shared.models import LineItem


def test_qty_prefix_stripped():
    assert fallback_canonical("2 X MILK 2L") == "MILK 2L"


def test_punctuation_removed():
    assert fallback_canonical("F&N MILK, 2L.") == "F&N MILK 2L"


def test_abbreviations_expanded():
    assert fallback_canonical("PKT ROTI") == "PACKET BREAD"
    assert fallback_canonical("BTL AIR") == "BOTTLE WATER"


def test_price_like_name_dropped():
    assert fallback_canonical("12.90") == ""
    assert fallback_canonical("RM 4.50") == ""


def test_empty_input():
    assert fallback_canonical("") == ""
    assert fallback_canonical("   ") == ""


def test_size_preserved_never_merged():
    """Different sizes are different products — never collapse them."""
    assert fallback_canonical("MILK 1L") != fallback_canonical("MILK 2L")


def test_case_and_spacing_normalised():
    assert fallback_canonical("  milk   2l  ") == "MILK 2L"


def test_align_uses_raw_key_when_present():
    pairs = [{"raw": "FN MILK 2L", "canonical": "F&N MILK 2L"},
             {"raw": "MILK 2L F&N", "canonical": "F&N MILK 2L"}]
    m = _align(["FN MILK 2L", "MILK 2L F&N"], pairs)
    assert m["FN MILK 2L"] == "F&N MILK 2L"
    assert m["MILK 2L F&N"] == "F&N MILK 2L"


def test_align_positional_when_no_raw_key():
    m = _align(["MILK 2L F&N", "ROTI"], ["F&N MILK 2L", "BREAD"])
    assert m["MILK 2L F&N"] == "F&N MILK 2L"
    assert m["ROTI"] == "BREAD"


def test_align_handles_short_response():
    """A truncated model reply must not erase names."""
    m = _align(["MILK", "ROTI"], [{"canonical": "MILK"}])
    assert m["MILK"] == "MILK"
    assert m["ROTI"] == "BREAD"          # ← fallback fills the gap


def test_canonical_key_prefers_canonical():
    assert canonical_key(
        LineItem("x", 1, 1.0, 1.0, canonical="MILK 2L")) == "MILK 2L"
    assert canonical_key(LineItem("PKT ROTI", 1, 1.0, 1.0)) == "PACKET BREAD"
```

### 3.29 `tests/test_ocr_engine.py`

```python
"""Locks the PaddleOCR output contract and bbox conversion."""

from src.extract.ocr_engine import (_to_bbox, box_str,
                                    low_confidence_lines, region_for_text)


def test_corner_points_to_bbox():
    """PaddleOCR gives 4 corner points → [x1, y1, x2, y2]."""
    box = [[10, 20], [110, 20], [110, 60], [10, 60]]
    assert _to_bbox(box) == [10, 20, 110, 60]


def test_bbox_handles_rotation():
    box = [[12, 18], [112, 22], [108, 62], [8, 58]]
    x1, y1, x2, y2 = _to_bbox(box)
    assert x1 == 8 and y1 == 18 and x2 == 112 and y2 == 62


def test_box_str_format():
    assert box_str([10, 20, 110, 60]) == "10,20,110,60"


def test_low_confidence_detection():
    regions = [{"confidence": 0.95, "text": "A"},
               {"confidence": 0.42, "text": "B"}]
    assert len(low_confidence_lines(regions)) == 1


def test_region_lookup_by_text():
    regions = [{"bbox_px": [0, 0, 10, 10], "text": "MILK 2L",
                "confidence": 0.9}]
    assert region_for_text(regions, "MILK") == "0,0,10,10"
    assert region_for_text(regions, "ROTI") == ""
```

### 3.30 `tests/test_check_total.py`

```python
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
```

### 3.31 `tests/test_check_price.py`

```python
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
```

### 3.32 `tests/test_check_duplicate.py`

```python
"""Locks the three-way split and within-receipt detection."""

from src.detect.check_duplicate import (check_within, classify_pair,
                                        MATCH, NON_MATCH)
from src.shared.models import Receipt, LineItem


def _r(items, vendor="Tesco", total=None, rid="a"):
    r = Receipt(vendor=vendor, date="2026-10-20", receipt_id=rid)
    r.items = items
    r.receipt_total = total if total is not None else r.subtotal()
    return r


def test_within_receipt_duplicate_detected():
    r = _r([LineItem("Milk", 1, 3.50, 3.50),
            LineItem("Milk", 1, 3.50, 3.50)])
    flags = check_within(r)
    assert len(flags) == 1
    assert flags[0].anomaly_type == "duplicate"


def test_different_qty_not_duplicate():
    r = _r([LineItem("Milk", 1, 3.50, 3.50),
            LineItem("Milk", 2, 3.50, 7.00)])
    assert check_within(r) == []


def test_identical_receipts_score_match():
    a = _r([LineItem("Milk", 1, 3.50, 3.50)], total=3.50, rid="a")
    b = _r([LineItem("Milk", 1, 3.50, 3.50)], total=3.50, rid="b")
    verdict, score = classify_pair(a, b)
    assert verdict == MATCH and score >= 0.9


def test_different_receipts_non_match():
    a = _r([LineItem("Milk", 1, 3.50, 3.50)], vendor="Tesco",
           total=3.50, rid="a")
    b = _r([LineItem("Rice", 1, 25.00, 25.00)], vendor="Mydin",
           total=25.00, rid="b")
    verdict, _ = classify_pair(a, b)
    assert verdict == NON_MATCH
```

### 3.33 `tests/test_explain.py`

```python
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
```

---

## PART 4 — WHAT ELSE YOU NEED, BEYOND DATASET SOURCES

### 4.1 Environment — light, and that is the win

| Requirement | Detail |
|---|---|
| **Python 3.9+** | PaddleOCR supports a wide range |
| **`paddleocr` + `paddlepaddle` (CPU)** | Installs in minutes. GPU build optional |
| **`openai` SDK** | DeepSeek uses the OpenAI-compatible endpoint |
| **DeepSeek API key** | Set in `.env`. **Run `health_check()` before the demo** |
| **No CUDA, no flash-attn, no multi-GB model download** | Nothing GPU-bound remains in the project |

**API budget:** three calls per receipt (scalars, items, names) plus optional polish. For ~150 receipts that is ~450 calls. **Cache them** — the cache is what makes your metrics reproducible, and it means the demo can run offline.

### 4.2 DeepSeek integration — the operational details

**Endpoint and settings** (all in `thresholds.yaml`):

```yaml
deepseek:
  model: deepseek-chat
  base_url: https://api.deepseek.com
  temperature: 0
  max_retries: 3
```

**Four things that will bite you:**

| Issue | Mitigation |
|---|---|
| **Rate limits / 429** | Retry with exponential backoff — already in `_call()` |
| **`json_object` forces an object** | Arrays must be wrapped. `expect_key="items"` unwraps it |
| **Truncated list replies** | `_align()` fills missing names from the local fallback instead of erasing them |
| **Live API during the demo** | **Never.** Replay cached JSON |

**The cache directory is not optional.** It sits in `src/extract/cache/`, is gitignored, and holds every model response. Run extraction once, freeze it, then run detection and metrics against the frozen file — that is what makes precision and recall stable numbers rather than a fresh roll of the dice each run.

### 4.3 Genuinely dataset-dependent — nothing you can code around

| Need | Depends on |
|---|---|
| **Reference pool construction** | A single receipt cannot tell whether a price is unusual — an external reference is mandatory. **Check 3 flags nothing until this exists** |
| **Per-product sample counts** | Your reliability rule requires ≥5–10 samples before a reference is trusted. **Most items in a 50-receipt set will not qualify** — that is expected |
| **FPR measurement** | Needs clean receipts that are **distinct** from the pool. If they overlap, your duplicate checker fires on your own clean data |
| **Promotion handling** | Promotions are in the test set, but no method distinguishes a promotional price from an anomaly |
| **Malaysian confirmation for any public source** | Your docs state this must be verified before the dataset choice is finalised |
| **Currency confirmation** | The candidate dataset's content audit lists name/qty/total_price/unit_price — no currency field. Your scope forbids pooling non-MYR prices |
| **Licensing / terms** | Confirm for each public dataset actually used |

### 4.4 The dataset contradiction still unresolved

**Your Primary dataset is SROIE**, but SROIE is described as *"company/date/address/total"* — **no per-item unit price**. That collides with two of your own rules: the inclusion test requires *"Has qty + unit_price (line items)"*, and *"the entire anomaly layer depends on itemized qty × unit_price"*.

**SROIE fails the filter it is meant to pass.** Either it is a detection-only benchmark, or another dataset supplies the itemized fields.

### 4.5 Decisions data alone will NOT settle

| # | Decision | Why data won't fix it |
|---|---|---|
| **B1** | Price reference source | Six options listed; none chosen. **Check 3 cannot run** |
| **B2** | Duplicate weights, thresholds, **history source** | Weights need labelled data *to tune* — but the history source must be chosen first, and cross-receipt detection is impossible without it. **`within_receipt` is the default and is unblocked today** |
| **B3** | Schema `discount` / `tax` | A design choice. Late change means reprocessing the dataset |
| **B5** | Metric definitions | The script gives human verification as a count; the deck lists a false-positive rate separately. Pick one and pin it |

### 4.6 OCR-specific open risks

| Issue | Status |
|---|---|
| **PP-Structure on curved thermal paper** | Receipts curl; the table parser assumes a flatter layout. Verify on your 20 photos |
| **Column-order recovery** | Varies by store. The layout hint helps, but the mapper can still mis-assign — the arithmetic gate is what catches it |
| **Low-confidence threshold (0.80)** | Unvalidated. Tune it or you will over-route to review |
| **Two failure modes** | PaddleOCR can misread; DeepSeek can mis-structure. The arithmetic gate catches both — which is why it is not optional |
| **Name normalisation drift** | If DeepSeek canonicalises inconsistently, Check 3's pool fragments again. `_name_source` in provenance tells you which path ran |

### 4.7 The three review triggers — a design point worth reporting

Your UI has **three** independent reasons a row goes to a human, and they catch different failures:

```
1. line arithmetic fails        →  a numeric error, or a mis-mapped column
2. OCR confidence below 0.80    →  the reader itself is unsure
3. mapper flagged a mismatch    →  the model saw an inconsistency
```

**Only the first is a check.** The other two are *triage* — they route work before detection runs. That distinction is worth making explicit in your report, because it explains why the review load is what it is.

### 4.8 Deck fixes — you are presenting MYR, the deck still shows USD

| # | Where | Fix |
|---|---|---|
| 1 | Title | "Receipts and Invoices" → **receipts only** |
| 2 | OCR example | Shows `USD` and a non-Malaysian store name → **MYR** |
| 3 | Check 1 example | Uses `$3 / $5 / $4`, total `$16` → **RM** |
| 4 | Receipt image example | `$64.42` amounts → **RM** |
| 5 | XAI example | Uses `$45.90` and `$32.00` → **RM** |
| 6 | Metrics | Add **F1** if the deck promises it |

### 4.9 Still open, straight from your own next-steps list

- Verify any public source is Malaysian — else find a Malaysian MYR source
- Decide the unusual-price reference source and **document it**
- Confirm dataset licensing
- Consider marking `tax` optional in the schema

### 4.10 Unresolved technical issues with no home yet

| Issue | Status |
|---|---|
| OCR errors propagate into duplicate matching | No mitigation designed |
| Same store/date/total ≠ proof of duplication | Unresolved |
| `MAD = 0` | **Guarded** — `modified_z` returns `None`; test added |
| **`confidence` field meaning and scale** | **Never defined in your docs.** The code invents formulas (`min(1, \|z\|/10)`, `\|diff\|/expected`). Document that they are yours |
| Minimum comparables: 4–5 vs ≥5–10 | Config uses 5; your docs say "≥5–10". Pick one |
| Human-verification metric unit | No unit defined |
| Rejected flags and the ledger | `export.py` drops them — confirm that is the intended policy |

---

## PART 5 — BUILD ORDER

**Phase 0 (one sitting, unblocks everything)**

1. `config/schema.py` + `thresholds.yaml` + `thresholds_loader.py` + `prompts.py`
2. Answer **B1, B2, B3, B5**
3. `src/shared/arithmetic.py`, `models.py`, `validators.py`, `deepseek_client.py`
4. `data/normal/` + `data/error/` + `data/reference/`
5. `pip install -r requirements.txt` — **verify on one receipt today**
6. Set `DEEPSEEK_API_KEY` and run `deepseek_client.health_check()`

**Then three lanes in parallel**

| Lane | First file | Why first |
|---|---|---|
| **A** (P3) | `ocr_engine.py` + its test | No model, no API — run it on one photo immediately |
| **B** (P4) | `check_total.py` | Simplest check, zero dataset dependency |
| **C** (P5) | `app.py` on mock JSON | Your gate requires mock-data export early |

**Lane A order:** `ocr_engine` → `field_mapper` → `name_normalizer` → `preprocessing` → accuracy work
**Lane B order:** `check_total` → `check_price` → `reference_builder` → `check_duplicate`
**Lane C order:** mock UI → `error_injector` → dataset → `compute_metrics`

**Smoke test before anything else:**

```bash
python -c "
from src.extract.ocr_engine import read
r = read('data/photos/test1.jpg', use_cache=False)
print(r['text'][:500])
print(f\"{len(r['regions'])} regions\")
print(r['regions'][0] if r['regions'] else 'none')
"
```

If that prints readable text and a `bbox_px` list, Lane A is unblocked.

**Second smoke test — the API, before you burn a batch:**

```bash
python -c "
from src.shared import deepseek_client as ds
print(ds.health_check())
print(ds.canonicalise(['PKT ROTI', '2 X MILK 2L', 'FN MILK 2 LTR']))
"
```

If that returns canonical names, step ④ works and Check 3 has a chance of having a usable pool.

---

## APPENDIX — SOURCES

| Content | Basis |
|---|---|
| Both JSON contracts, four-field anomaly contract | `DocAudit_Proposal_MYR.md` §8 |
| Workflow, checks, formulas, gates, metrics | markdown_193 §5, §7, §11–13; markdown_180 §5–6 |
| Malin: SST embedded in shelf price | markdown_191 §3 |
| SROIE field mismatch and open Malaysian status | markdown_180 §8.4; markdown_191 §5 |
| Explanation must give reason + supporting source content | `DocAudit_Project_Proposal.pdf` |
| **All file contents, all `confidence` formulas, the three review triggers, `reference_builder.py`, `name_normalizer.py`, `export.py`, `app.py`** | **Proposed here** — required by the design, named in no source |
