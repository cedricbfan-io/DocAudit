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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extract.ocr_engine import run_ocr, read_score, needs_chinese
from config.thresholds_loader import THRESHOLDS
from src.extract.preprocessing import preprocess
from src.extract import receipt_parser
from src.extract.name_normalizer import fallback_canonical
from src.extract.fallback_extract import _parse_date, _find_date, _find_vendor
from src.shared.arithmetic import line_ok, receipt_ok, is_vacuous
from src.shared.models import Receipt, LineItem
from src.shared.validators import validate_receipt
from src.shared import deepseek_client
from config.thresholds_loader import THRESHOLDS

_REVIEW_BELOW = THRESHOLDS["ocr"]["confidence"]["review_below"]


_NAME_BLACKLIST = {
    'SR', 'SST', 'GST', 'T6', 'T6%', 'SR 6', 'SR6', 'SR06',
    'S 6', 'Z 0', 'S 6/', '6 ST', '6%', '@6', 'CODE',
    'TABLE', 'ORDERNO', 'ITEM', 'PAID', 'PAYMENT',
    'RM', 'CAL', 'ZRL', 'HSUO', 'TLRM', 'TOTA',
    'NO OF ITEMS', 'ITEM COUNT', 'ITEN COUNT', 'MAIN ITEM',
    'PEKING FEE', 'TOEEL', 'NET AMT', 'SUBTETA',
}


def structure_regions(regions, receipt_id="", image_path="", clean="",
                      header=None, mapping_source="rules"):
    """OCR regions → Receipt. Pure function: no OCR, no network.

    Split out of extract_hybrid so it can be unit-tested on recorded
    or synthetic regions (tests/test_receipt_parser.py).
    `header` may supply vendor/date/receipt_total from the API or SROIE
    ground truth; anything missing is read from the regions by rules.
    """
    header = header or {}
    regions = sorted(regions, key=lambda r: (round(r["bbox_px"][1] / 10),
                                             r["bbox_px"][0]))
    text_lines = [r["text"] for r in regions]

    totals = receipt_parser.find_totals(regions)
    vendor = header.get("vendor") or _find_vendor(text_lines, regions)
    date = header.get("date") or _find_date(text_lines)
    receipt_total = header.get("receipt_total")
    if receipt_total is None:
        receipt_total = totals["total"]

    tax, tax_mode = _tax_mode(totals)

    unparsed = []
    derived = []
    items = []
    for it in receipt_parser.parse_items(regions, unparsed_out=unparsed):
        if it["line_total"] < 0 or it["unit_price"] < 0:
            continue                      # discount / refund rows
        canonical = fallback_canonical(it["name"])
        if not canonical or canonical in _NAME_BLACKLIST:
            continue
        items.append(LineItem(
            name=it["name"], qty=it["qty"], unit_price=it["unit_price"],
            line_total=it["line_total"], canonical=canonical,
            raw_text=it["raw_text"], region=it["region"],
            ocr_confidence=it["confidence"],
        ))
        if it.get("derived"):
            # observed string stays in raw_text; the split is a derived value
            derived.append({"row": len(items) - 1, "note": it["derived"],
                            "raw_text": it["raw_text"]})

    receipt = Receipt(
        vendor=vendor,
        date=date,
        currency=_detect_currency(text_lines),
        items=items,
        receipt_total=receipt_total,
        tax=tax,
        printed_subtotal=totals["subtotal"],
        receipt_id=receipt_id,
        image_path=image_path,
        regions=regions,
        provenance={
            "_reader": "paddleocr",
            "_mapping": mapping_source,
            "_clean_image": clean,
            "_name_source": "fallback",
            "_parser": "receipt_parser",
            "_total_region": totals["total_region"],
            "_subtotal_region": totals["subtotal_region"],
        },
    )
    if totals.get("tax") is not None:
        # observed printed value(s); `receipt.tax` holds it only when it
        # is added on top of the subtotal
        receipt.provenance["_tax"] = {
            "printed": totals["tax"], "mode": tax_mode, "added": tax,
            "region": totals["tax_region"], "rows": totals["tax_rows"]}
    if receipt.receipt_total is None and items:
        bottoms = [float(it.region.split(",")[3]) for it in items
                   if it.region and it.region.count(",") == 3]
        inf = receipt_parser.infer_total(regions, receipt.subtotal(),
                                         below_y=max(bottoms, default=0))
        if inf:
            # observed value from that row; the reason it was chosen is kept
            receipt.receipt_total = inf["total"]
            receipt.provenance["_total_region"] = inf["total_region"]
            receipt.provenance["_total_inferred"] = {
                "row": inf["raw_text"], "items_sum": receipt.subtotal(),
                "tax": inf["tax"]}
            if inf["tax"]:
                receipt.tax = inf["tax"]
    if unparsed:
        receipt.provenance["_unparsed_rows"] = unparsed
    if derived:
        receipt.provenance["_derived_fields"] = derived

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
            ok, expected, diff = receipt_ok(subtotal, 0.0, receipt.tax or 0.0,
                                             receipt.receipt_total)
            if not ok:
                receipt.provenance["_total_gap"] = {
                    "expected": expected,
                    "observed": receipt.receipt_total,
                    "difference": diff,
                }

    validate_receipt(receipt)
    return receipt


def _detect_currency(lines):
    """Currency from what is printed; MYR (the project's scope) by default.

    ¥ / ￥ / 元 / RMB / Chinese text → CNY; '$' with no RM → USD.
    """
    text = " ".join(lines)
    if re.search(r"\bRM\s?\d|\bMYR\b|\bRM\b", text):
        return "MYR"
    if re.search(r"[¥￥元]|\bRMB\b|\bCNY\b", text) or \
            len(re.findall(r"[\u4e00-\u9fff]", text)) >= 8:
        return "CNY"
    if re.search(r"\bS\$", text):
        return "SGD"
    if "$" in text or re.search(r"\bUSD\b", text):
        return "USD"
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    return "MYR"


def _tax_mode(totals):
    """(amount added on top of the items, mode) from PRINTED rows only.

    'exclusive'    — a printed SUBTOTAL differs from the total, so tax /
                     service charge printed above the total is added:
                     S + T = X.
    'inclusive'    — the printed subtotal already equals the total; the
                     tax line only states the tax inside the prices.
    'service_only' — no printed subtotal: only a service charge is added
                     (it is never inside item prices); GST/SST may be
                     inside the prices (SROIE train_105), so it is not.
    'unknown'      — no printed subtotal and no service charge.
    The line items are never used here, so a wrong total cannot make the
    tax disappear.
    """
    tax = totals.get("tax")
    if tax is None:
        return None, None
    sub, total = totals.get("subtotal"), totals.get("total")
    if sub is not None and total is not None:
        if abs(sub - total) <= 0.005:
            return None, "inclusive"
        return tax, "exclusive"
    if totals.get("service_charge"):
        return totals["service_charge"], "service_only"
    return None, "unknown"


def extract_hybrid(image_path, sroie_gt=None, use_cache=True, skip_api=False,
                   ocr=None):
    """Photo → Receipt.

    PaddleOCR reads; receipt_parser structures (layout rules, no LLM).
    Header fields come from SROIE ground truth (eval), the DeepSeek API,
    or — with skip_api=True, as the demo UI uses — from rules only.
    """
    rid = Path(image_path).stem
    # `ocr(path, lang)` is replaceable so the evaluation scripts run this
    # exact pipeline on cached OCR output (eval/real_world.py)
    ocr = ocr or (lambda path, lang=None: run_ocr(path, lang=lang))
    clean = preprocess(image_path)
    ocr_lang = THRESHOLDS["ocr"]["lang"]
    regions = ocr(clean, None)

    header, mapping_source = {}, "rules"
    if sroie_gt:
        header["vendor"] = sroie_gt.get("company")
        header["date"] = _parse_date(sroie_gt.get("date"))
        total_str = sroie_gt.get("total")
        if total_str:
            try:
                header["receipt_total"] = float(total_str.replace(',', ''))
            except ValueError:
                pass
        mapping_source = "sroie_gt"
    elif not skip_api:
        try:
            full_text = "\n".join(r["text"] for r in regions)
            scalars = deepseek_client.map_scalars(full_text, regions,
                                                   use_cache=use_cache)
            header = {k: scalars.get(k) for k in
                      ("vendor", "date", "receipt_total")}
            mapping_source = "nvidia"
        except Exception as e:
            mapping_source = f"rules_fallback:{e}"

    receipt = structure_regions(regions, receipt_id=rid, image_path=image_path,
                                clean=clean, header=header,
                                mapping_source=mapping_source)
    # The English model turns Chinese into confident-looking junk, so a
    # run that found no total is retried with the
    # Chinese model (which also reads English and digits); the run that
    # recovered more structure is kept.
    fallback = THRESHOLDS["ocr"].get("fallback_lang")
    if fallback and ocr_lang != fallback and (
            receipt.receipt_total is None or needs_chinese(regions)):
        try:
            zh = ocr(clean, fallback)
        except Exception:                    # noqa: BLE001
            zh = None
        if zh:
            alt = structure_regions(zh, receipt_id=rid, image_path=image_path,
                                    clean=clean, header=header,
                                    mapping_source=mapping_source)
            better = ((alt.receipt_total is not None) - (receipt.receipt_total is not None),
                      len(alt.items) - len(receipt.items),
                      read_score(zh) - read_score(regions))
            if better > (0, 0, 0):
                receipt, ocr_lang = alt, fallback
    receipt.provenance["_ocr_lang"] = ocr_lang
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
