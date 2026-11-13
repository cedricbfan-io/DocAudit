"""★ GATE DELIVERABLE: extract.py outputs valid schema-compliant JSON.

Gate: detection works on ≥15/20 photos (≥75%); clean-photo field-level
accuracy ≥85%.
"""

import json
import sys
from pathlib import Path

from config.thresholds_loader import THRESHOLDS
from src.extract import field_mapper, name_normalizer, ocr_engine
from src.extract.preprocessing import preprocess
from src.shared.arithmetic import line_ok, receipt_ok, is_vacuous
from src.shared.models import Receipt
from src.shared.validators import validate_receipt

_REVIEW_BELOW = THRESHOLDS["ocr"]["confidence"]["review_below"]


def run_arithmetic_gate(receipt: Receipt) -> Receipt:
    """Step ⑤ — the human-work killer.

    Only failures become review_required. Everything else stays
    'unreviewed' and is auto-accepted. This is what makes the reader
    choice safe: a misread breaks the arithmetic and announces itself.

    THREE independent review triggers:
      1. line arithmetic fails      (qty × unit_price ≠ line_total)
      2. OCR confidence below floor
      3. the mapper itself flagged a mismatch
    """
    for item in receipt.items:
        reasons = []
        if not line_ok(item.qty, item.unit_price, item.line_total):
            reasons.append("line arithmetic")
        if item.ocr_confidence < _REVIEW_BELOW:
            reasons.append(f"low OCR confidence ({item.ocr_confidence:.2f})")
        if item.mismatch:
            reasons.append("mapper flagged mismatch")

        if reasons:
            item.review = "review_required"
            item.reason = "; ".join(reasons)

    if receipt.receipt_total is not None:
        subtotal = receipt.subtotal()
        tax = receipt.tax or 0.0
        if is_vacuous(subtotal, receipt.receipt_total):
            receipt.provenance["_vacuous"] = True
        else:
            ok, expected, diff = receipt_ok(
                subtotal, 0.0, tax, receipt.receipt_total)
            if not ok:
                receipt.provenance["_total_gap"] = {
                    "expected": expected,
                    "observed": receipt.receipt_total,
                    "difference": diff,
                }
    return receipt


def extract_one(image_path: str, receipt_id: str = "",
                use_cache: bool = True) -> Receipt:
    """Photo → schema-compliant Receipt. The full ①–⑤ sequence."""
    rid = receipt_id or Path(image_path).stem

    clean = preprocess(image_path)                              # ①
    ocr = ocr_engine.read(clean, use_cache=use_cache)           # ②
    receipt = field_mapper.map_receipt(                          # ③
        ocr, rid, image_path, use_cache=use_cache)
    receipt.provenance["_clean_image"] = clean

    if not field_mapper.validate_currency(receipt):
        raise ValueError(f"non-MYR receipt rejected: {image_path}")

    receipt = name_normalizer.normalise_receipt(                 # ④
        receipt, use_cache=use_cache)

    receipt = run_arithmetic_gate(receipt)                       # ⑤
    validate_receipt(receipt)
    return receipt


def to_json(receipt: Receipt) -> dict:
    """Emit the frozen line-item contract, plus provenance."""
    return {
        "receipt_id": receipt.receipt_id,
        "vendor": receipt.vendor,
        "date": receipt.date,
        "currency": receipt.currency,
        "items": [
            {"name": i.name, "canonical": i.canonical,
             "qty": i.qty, "unit_price": i.unit_price,
             "line_total": i.line_total, "review": i.review,
             "reason": i.reason, "region": i.region,
             "ocr_confidence": round(i.ocr_confidence, 4)}
            for i in receipt.items
        ],
        "receipt_total": receipt.receipt_total,
        "printed_subtotal": receipt.printed_subtotal,
        "tax": receipt.tax,
        "duplicate_flag": receipt.duplicate_flag,
        "_provenance": receipt.provenance,
        "_needs_review": receipt.needs_review(),
    }


def extract_batch(image_paths: list, use_cache: bool = True) -> list:
    """Never kill a whole batch on one bad receipt."""
    out = []
    for p in image_paths:
        try:
            out.append(to_json(extract_one(p, use_cache=use_cache)))
        except Exception as e:
            out.append({"error": str(e), "image": p})
    return out


def save_batch(image_paths: list, out_dir: str = "data/extracted") -> str:
    """Cache a whole batch to disk — this is what freezes your metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = extract_batch(image_paths)
    (out / "extracted.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out / "extracted.json")


if __name__ == "__main__":
    print(json.dumps(extract_batch(sys.argv[1:]), indent=2,
                     ensure_ascii=False))
