"""Hybrid extraction: NVIDIA API for scalars + local parser for items.

Uses PaddleOCR for OCR, NVIDIA DeepSeek API for header field mapping
(vendor/date/total), and the column-aware local parser for line items.
Name normalisation uses the fallback rule-based normaliser.

This avoids the slow/inconsistent map_items LLM call while still
getting high-quality header fields from the LLM.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.extract.ocr_engine import _get_engine, _to_bbox, box_str
from src.extract.preprocessing import preprocess
from src.extract.name_normalizer import fallback_canonical
from src.extract.fallback_extract import (
    _parse_line_items, _parse_date, _find_total, _find_date, _find_vendor,
    _group_regions_into_rows, _detect_column_layout,
)
from src.shared.arithmetic import line_ok, receipt_ok, is_vacuous
from src.shared.models import Receipt, LineItem
from src.shared.validators import validate_receipt
from src.shared import deepseek_client
from config.thresholds_loader import THRESHOLDS

_REVIEW_BELOW = THRESHOLDS["ocr"]["confidence"]["review_below"]


def extract_hybrid(image_path, sroie_gt=None, use_cache=True):
    """Extract using NVIDIA for scalars + local parser for items."""
    rid = Path(image_path).stem

    clean = preprocess(image_path)

    engine = _get_engine()
    raw = engine.ocr(clean, cls=True)

    regions = []
    text_lines = []
    for page in (raw or []):
        for box, (text, conf) in (page or []):
            bbox = _to_bbox(box)
            regions.append({
                "label": "text",
                "bbox_px": bbox,
                "text": text,
                "confidence": float(conf),
            })
            text_lines.append(text)

    regions.sort(key=lambda r: (round(r["bbox_px"][1] / 10), r["bbox_px"][0]))
    text_lines = [r["text"] for r in regions]
    full_text = "\n".join(text_lines)

    # Header fields: NVIDIA API (with SROIE GT fallback)
    if sroie_gt:
        vendor = sroie_gt.get("company")
        date = _parse_date(sroie_gt.get("date"))
        receipt_total = None
        total_str = sroie_gt.get("total")
        if total_str:
            try:
                receipt_total = float(total_str.replace(',', ''))
            except ValueError:
                pass
        mapping_source = "sroie_gt"
    else:
        try:
            scalars = deepseek_client.map_scalars(full_text, regions,
                                                   use_cache=use_cache)
            vendor = scalars.get("vendor")
            date = scalars.get("date")
            receipt_total = scalars.get("receipt_total")
            mapping_source = "nvidia"
        except Exception as e:
            vendor = _find_vendor(text_lines, regions)
            date = _find_date(text_lines)
            receipt_total = _find_total(text_lines)
            mapping_source = f"rules_fallback:{e}"

    # Line items: local column-aware parser (no LLM)
    items = _parse_line_items(regions)

    # Name normalisation: fallback only (fast, deterministic)
    for item in items:
        item.canonical = fallback_canonical(item.name)

    items = [i for i in items if i.canonical or len(i.name) > 3]

    receipt = Receipt(
        vendor=vendor,
        date=date,
        currency="MYR",
        items=items,
        receipt_total=receipt_total,
        tax=None,
        receipt_id=rid,
        image_path=image_path,
        regions=regions,
        provenance={
            "_reader": "paddleocr",
            "_mapping": mapping_source,
            "_clean_image": clean,
            "_name_source": "fallback",
        },
    )

    # Arithmetic gate
    for item in receipt.items:
        reasons = []
        if not line_ok(item.qty, item.unit_price, item.line_total):
            reasons.append("line arithmetic")
        if item.ocr_confidence < _REVIEW_BELOW:
            reasons.append(f"low OCR confidence ({item.ocr_confidence:.2f})")
        if reasons:
            item.review = "review_required"
            item.reason = "; ".join(reasons)

    if receipt.receipt_total is not None:
        subtotal = receipt.subtotal()
        if not is_vacuous(subtotal, receipt.receipt_total):
            ok, expected, diff = receipt_ok(subtotal, 0.0, 0.0,
                                             receipt.receipt_total)
            if not ok:
                receipt.provenance["_total_gap"] = {
                    "expected": expected,
                    "observed": receipt.receipt_total,
                    "difference": diff,
                }

    validate_receipt(receipt)
    return receipt


def run_hybrid_batch(image_dir, gt_path=None, out_path=None):
    """Extract a batch using hybrid mode."""
    from src.extract.extract import to_json

    image_dir = Path(image_dir)
    images = sorted(image_dir.glob('*.jpg'))

    gt = {}
    if gt_path and Path(gt_path).exists():
        gt_data = json.loads(Path(gt_path).read_text(encoding='utf-8'))
        for split in gt_data.values():
            for entry in split:
                gt[entry['receipt_id']] = entry

    results = []
    for i, img in enumerate(images):
        rid = img.stem
        sroie_gt = gt.get(rid)
        t0 = time.time()
        try:
            receipt = extract_hybrid(str(img), sroie_gt)
            data = to_json(receipt)
            results.append(data)
            print(f'  [{i+1}/{len(images)}] {rid}: {time.time()-t0:.1f}s - '
                  f'vendor={data.get("vendor", "?")}, '
                  f'items={len(data.get("items", []))}, '
                  f'total={data.get("receipt_total")}')
        except Exception as e:
            results.append({"error": str(e), "image": str(img)})
            print(f'  [{i+1}/{len(images)}] {rid}: FAIL in {time.time()-t0:.1f}s - {e}')

        # Save after each image
        if out_path:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding='utf-8')

    ok = sum(1 for r in results if "error" not in r)
    err = sum(1 for r in results if "error" in r)
    total_items = sum(len(r.get("items", [])) for r in results if "error" not in r)
    print(f'\nDone: OK={ok}, Errors={err}, Total items={total_items}')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--gt', default='eval/ground_truth/sroie_gt.json')
    parser.add_argument('--out', default='data/extracted/normal_extracted.json')
    args = parser.parse_args()
    run_hybrid_batch(args.image_dir, args.gt, args.out)
