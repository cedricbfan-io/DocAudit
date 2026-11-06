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
from src.extract.extract import to_json
from src.extract.hybrid_extract import extract_hybrid
from src.shared.models import Receipt, LineItem
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
        receipt = extract_hybrid(image_path)
    except Exception as e:
        return f"Extraction failed: {e}", None, "", None

    pool = _load_pool()
    result = Detector(price_pool=pool).run(receipt)

    data = to_json(receipt)
    _STATE["receipt"] = data
    _STATE["result"] = result
    _STATE["decisions"] = {}

    flagged = [i["region"] for i in data["items"] if i["reason"]]
    ann_image = receipt.provenance.get("_clean_image", image_path)
    ann = highlight.annotate(ann_image, receipt.regions, flagged) \
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


def _receipt_from_json(data: dict) -> Receipt:
    """Reconstruct a Receipt from extracted JSON (for error-injected cases)."""
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
        tax=data.get("tax"), receipt_id=data.get("receipt_id", ""),
        image_path=data.get("_provenance", {}).get("_clean_image", ""),
        regions=[], provenance=data.get("_provenance", {}),
    )


def process_json(json_path: str):
    """Load a pre-extracted receipt JSON and run detection."""
    if not json_path:
        return "Upload a JSON file first.", None, "", None

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as e:
        return f"JSON load failed: {e}", None, "", None

    receipt = _receipt_from_json(data)
    pool = _load_pool()
    result = Detector(price_pool=pool).run(receipt)

    _STATE["receipt"] = data
    _STATE["result"] = result
    _STATE["decisions"] = {}

    table = [[i["name"], i.get("canonical", ""), i["qty"], i["unit_price"],
              i["line_total"], i.get("review", ""), i.get("reason", "")]
             for i in data.get("items", [])]

    stats = pool_stats(pool)
    header = (f"**{data.get('vendor') or 'Unknown vendor'}** · "
              f"{data.get('date') or 'no date'} · "
              f"Total RM {data.get('receipt_total') or 0:.2f}\n\n"
              f"{result['summary']}\n\n"
              f"Price pool: {stats['items']} items "
              f"({stats['observations']} observations)")

    anomalies = "\n".join(
        f"**{a['anomaly_type']}** ({a['confidence']}) — {a['reason']}"
        for a in result["anomalies"]) or "No anomalies detected."

    return header, None, anomalies, table


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
    return "LLM OK" if h["ok"] else f"LLM FAILED: {h['error']}"


def build_ui():
    with gr.Blocks(title="DocAudit — MYR Receipt Anomaly Detection") as demo:
        gr.Markdown("# DocAudit — MYR Receipt Anomaly Detection\n"
                    "Read with PaddleOCR · Structure with LLM · "
                    "Decide with code · Explain with templates")

        with gr.Tabs():
            with gr.Tab("Image (full pipeline)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(type="filepath", label="Receipt photo")
                        run_btn = gr.Button("Analyse", variant="primary")
                        health_btn = gr.Button("LLM health check")
                        health_out = gr.Textbox(label="API status", lines=2)

                    with gr.Column(scale=2):
                        header_out = gr.Markdown()
                        img_out = gr.Image(label="Evidence regions", type="pil")
                        anomalies_out = gr.Markdown(label="Anomalies")

            with gr.Tab("JSON (pre-extracted / error-injected)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        json_in = gr.File(label="Receipt JSON",
                                          file_types=[".json"])
                        json_btn = gr.Button("Run detection", variant="primary")
                        gr.Markdown(
                            "Load error-injected receipts from "
                            "`eval/error_set/receipts/` to test detection.")

                    with gr.Column(scale=2):
                        json_header = gr.Markdown()
                        json_anomalies = gr.Markdown(label="Anomalies")

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
        json_btn.click(process_json, inputs=json_in,
                       outputs=[json_header, json_in, json_anomalies, table_out])
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
