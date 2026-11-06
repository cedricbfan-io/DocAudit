"""Fallback extraction: PaddleOCR + SROIE ground truth (no DeepSeek needed).

Uses PaddleOCR for OCR regions/bboxes, and SROIE ground-truth entities
for header fields. Attempts rule-based line-item parsing from OCR text.

This bypasses DeepSeek entirely — the pipeline still runs the arithmetic
gate, name normalisation (fallback), and all three detection checks.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extract.ocr_engine import _get_engine, _to_bbox, box_str, layout_hint
from src.extract.name_normalizer import fallback_canonical, canonical_key
from src.extract.preprocessing import preprocess
from src.shared.arithmetic import line_ok, receipt_ok, is_vacuous
from src.shared.models import Receipt, LineItem, Region
from src.shared.validators import validate_receipt
from config.thresholds_loader import THRESHOLDS

_REVIEW_BELOW = THRESHOLDS["ocr"]["confidence"]["review_below"]

# Patterns for header field extraction from OCR text
_TOTAL_RE = re.compile(r'(?:TOTAL|JUMLAH|AMOUNT|GRAND TOTAL)\s*RM?\s*([\d,.]+)',
                        re.IGNORECASE)
_DATE_RE = re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')
_PRICE_RE = re.compile(r'RM?\s*([\d]+\.\d{2})')
_QTY_RE = re.compile(r'^(\d+)\s*[xX*]\s*(.+)', re.IGNORECASE)


def _parse_date(raw):
    """Convert DD/MM/YYYY → YYYY-MM-DD."""
    if not raw:
        return None
    m = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', raw)
    if not m:
        return raw
    d, mo, y = m.groups()
    if len(y) == 2:
        y = '20' + y
    return f'{y}-{int(mo):02d}-{int(d):02d}'


def _find_total(text_lines):
    for line in reversed(text_lines):
        m = _TOTAL_RE.search(line)
        if m:
            val = m.group(1).replace(',', '')
            try:
                return float(val)
            except ValueError:
                pass
    return None


def _find_date(text_lines):
    for line in text_lines:
        m = _DATE_RE.search(line)
        if m:
            return _parse_date(m.group(1))
    return None


def _find_vendor(text_lines, regions):
    """Vendor is usually the first non-trivial text line near the top."""
    for r in regions:
        text = r.get("text", "").strip()
        if len(text) > 3 and not _PRICE_RE.match(text) and not _DATE_RE.match(text):
            return text
    return None


def _group_regions_into_rows(regions, y_tolerance=15):
    """Group OCR regions into rows by y-position proximity.

    Returns a list of rows, each row is a list of (x, text, conf, bbox) tuples
    sorted by x-position (left to right).
    """
    if not regions:
        return []

    # Sort by y, then x
    sorted_regions = sorted(regions, key=lambda r: (r["bbox_px"][1], r["bbox_px"][0]))

    rows = []
    current_row = []
    current_y = None

    for r in sorted_regions:
        y = r["bbox_px"][1]
        text = r.get("text", "").strip()
        if not text:
            continue
        x = r["bbox_px"][0]
        conf = r.get("confidence", 1.0)
        bbox = r["bbox_px"]

        if current_y is None or abs(y - current_y) <= y_tolerance:
            current_row.append((x, text, conf, bbox))
            current_y = y if current_y is None else current_y
        else:
            if current_row:
                current_row.sort(key=lambda t: t[0])
                rows.append(current_row)
            current_row = [(x, text, conf, bbox)]
            current_y = y

    if current_row:
        current_row.sort(key=lambda t: t[0])
        rows.append(current_row)

    return rows


def _detect_column_layout(rows):
    """Find the header row and detect column x-positions.

    Returns (col_x_ranges, header_row_index) or (None, None) if no header found.
    col_x_ranges is a dict: {'name': (x_min, x_max), 'qty': ..., 'price': ..., 'amount': ...}
    """
    for i, row in enumerate(rows):
        texts = [t.lower() for _, t, _, _ in row]
        joined = " ".join(texts)
        has_desc = any("desc" in t or "item" in t for t in texts)
        has_qty = any("qty" in t or "quantity" in t for t in texts)
        has_price = any("price" in t or "unit" in t for t in texts)
        has_amount = any("amount" in t or "total" in t for t in texts)

        if has_desc and (has_qty or has_price or has_amount):
            cols = {}
            for x, text, _, _ in row:
                tl = text.lower()
                if "desc" in tl or "item" in tl:
                    cols["name"] = (x, x + 200)
                elif "qty" in tl or "quantity" in tl:
                    cols["qty"] = (x, x + 50)
                elif "price" in tl or "unit" in tl:
                    cols["price"] = (x, x + 80)
                elif "amount" in tl or "total" in tl:
                    cols["amount"] = (x, x + 80)
            return cols, i

    return None, None


def _parse_line_items(regions):
    """Column-aware line-item parser.

    Groups OCR regions into rows, detects the column header, then parses
    each subsequent row as a line item using column positions.
    """
    rows = _group_regions_into_rows(regions)
    if not rows:
        return []

    cols, header_idx = _detect_column_layout(rows)

    # Skip keywords for non-item rows
    skip_kw = {'TOTAL', 'SUBTOTAL', 'CASH', 'CHANGE', 'TAX', 'SST', 'GST',
               'TENDER', 'BALANCE', 'JUMLAH', 'BAKI', 'TUNAI', 'DISCOUNT',
               'DISKAUN', 'VISA', 'MASTER', 'DEBIT', 'CREDIT', 'SIGNATURE',
               'THANK', 'CASHIER', 'DATE', 'CASH BILL', 'GOODS', 'PLEASE',
               'REFUNDABLE', 'AMOUNT', 'TOTAL:', 'TOTAL AMOUNT',
               'CASH RECEIVED', 'CHANGE:'}

    items = []

    if cols:
        # Column-aware mode: use header positions to assign columns
        start_row = header_idx + 1 if header_idx is not None else 0
        for row in rows[start_row:]:
            row_texts = [t for _, t, _, _ in row]
            row_upper = " ".join(row_texts).upper()
            if any(kw in row_upper for kw in skip_kw):
                continue

            name_parts = []
            qty_val = None
            price_val = None
            amount_val = None
            name_bbox = None

            for x, text, conf, bbox in row:
                # Assign to column based on x-position
                assigned = False
                for col_name, (x_min, x_max) in cols.items():
                    if x_min - 30 <= x <= x_max + 30:
                        if col_name == "name":
                            name_parts.append(text)
                            if name_bbox is None:
                                name_bbox = bbox
                        elif col_name == "qty":
                            try:
                                qty_val = float(text)
                            except ValueError:
                                pass
                        elif col_name == "price":
                            try:
                                price_val = float(text.replace(',', '').replace('RM', '').strip())
                            except ValueError:
                                pass
                        elif col_name == "amount":
                            try:
                                amount_val = float(text.replace(',', '').replace('RM', '').strip())
                            except ValueError:
                                pass
                        assigned = True
                        break

                if not assigned:
                    # Unassigned text — if it looks like a number, treat as amount
                    try:
                        num = float(text.replace(',', '').replace('RM', '').strip())
                        if amount_val is None:
                            amount_val = num
                        elif price_val is None:
                            price_val = num
                    except ValueError:
                        if x < 200:  # left side → probably name
                            name_parts.append(text)

            name = " ".join(name_parts).strip()
            if not name or len(name) < 2:
                continue
            if amount_val is None and price_val is not None:
                amount_val = price_val
            if price_val is None and amount_val is not None and qty_val:
                price_val = round(amount_val / qty_val, 2)
            if amount_val is None:
                continue
            if qty_val is None:
                qty_val = 1.0
            if price_val is None:
                price_val = amount_val

            region_str = box_str(name_bbox) if name_bbox else ""
            items.append(LineItem(
                name=name,
                qty=qty_val,
                unit_price=price_val,
                line_total=amount_val,
                raw_text=" ".join(row_texts),
                region=region_str,
                ocr_confidence=min(c for _, _, c, _ in row) if row else 0.85,
            ))
    else:
        # Fallback: no column header detected — try inline price parsing
        for row in rows:
            row_texts = [t for _, t, _, _ in row]
            joined = " ".join(row_texts)
            upper = joined.upper()
            if any(kw in upper for kw in skip_kw):
                continue

            prices = _PRICE_RE.findall(joined)
            if not prices:
                continue

            try:
                line_total = float(prices[-1].replace(',', ''))
            except ValueError:
                continue
            if line_total <= 0 or line_total > 9999:
                continue

            name_part = joined[:joined.rfind(prices[-1])].strip()
            name_part = re.sub(r'RM?\s*$', '', name_part).strip()
            if not name_part or len(name_part) < 2:
                continue

            qty = 1.0
            unit_price = line_total
            qm = _QTY_RE.match(name_part)
            if qm:
                qty = float(qm.group(1))
                name_part = qm.group(2).strip()
                if qty > 0:
                    unit_price = round(line_total / qty, 2)

            region_str = ""
            for r in regions:
                if name_part.lower() in (r.get("text") or "").lower():
                    region_str = box_str(r.get("bbox_px", []))
                    break

            items.append(LineItem(
                name=name_part,
                qty=qty,
                unit_price=unit_price,
                line_total=line_total,
                raw_text=joined,
                region=region_str,
                ocr_confidence=0.85,
            ))

    return items


def extract_with_paddleocr(image_path, sroie_gt=None):
    """Extract a receipt using PaddleOCR + optional SROIE ground truth.

    If sroie_gt is provided (dict with company/date/total/address),
    those values override the rule-based extraction for header fields.
    """
    rid = Path(image_path).stem

    # Preprocess
    try:
        clean = preprocess(image_path)
    except Exception:
        clean = image_path

    # OCR
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

    # Reading order: top-to-bottom, then left-to-right
    regions.sort(key=lambda r: (round(r["bbox_px"][1] / 10), r["bbox_px"][0]))
    text_lines = [r["text"] for r in regions]

    # Header fields: use SROIE GT if available, else rule-based
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
    else:
        vendor = _find_vendor(text_lines, regions)
        date = _find_date(text_lines)
        receipt_total = _find_total(text_lines)

    # Line items: always rule-based (SROIE has no line-item GT)
    items = _parse_line_items(regions)

    # Name normalisation (fallback only — no DeepSeek)
    for item in items:
        item.canonical = fallback_canonical(item.name)

    # Drop unusable items
    items = [i for i in items if i.canonical or len(i.name) > 3]

    # Build Receipt
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
        provenance={"_reader": "paddleocr", "_mapping": "sroie_gt" if sroie_gt else "rules",
                    "_clean_image": clean, "_name_source": "fallback"},
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
            ok, expected, diff = receipt_ok(subtotal, 0.0, 0.0, receipt.receipt_total)
            if not ok:
                receipt.provenance["_total_gap"] = {
                    "expected": expected,
                    "observed": receipt.receipt_total,
                    "difference": diff,
                }

    validate_receipt(receipt)
    return receipt


def extract_batch_sroie(image_dir, gt_path=None):
    """Extract a batch, optionally using SROIE ground truth."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob('*.jpg'))

    gt = {}
    if gt_path and Path(gt_path).exists():
        gt_data = json.loads(Path(gt_path).read_text(encoding='utf-8'))
        for split in gt_data.values():
            for entry in split:
                gt[entry['receipt_id']] = entry

    results = []
    for img in images:
        rid = img.stem
        sroie_gt = gt.get(rid)
        try:
            receipt = extract_with_paddleocr(str(img), sroie_gt)
            from src.extract.extract import to_json
            data = to_json(receipt)
            results.append(data)
            print(f'  {rid}: vendor={data.get("vendor", "?")}, '
                  f'items={len(data.get("items", []))}, '
                  f'total={data.get("receipt_total")}, '
                  f'review={data.get("_needs_review")}')
        except Exception as e:
            print(f'  {rid}: FAIL - {e}')
            results.append({"error": str(e), "image": str(img)})

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='Directory of receipt images')
    parser.add_argument('--gt', default='eval/ground_truth/sroie_gt.json',
                        help='SROIE ground truth JSON')
    parser.add_argument('--out', default='data/extracted/batch.json',
                        help='Output JSON path')
    args = parser.parse_args()

    results = extract_batch_sroie(args.image_dir, args.gt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved {len(results)} results to {out}')
