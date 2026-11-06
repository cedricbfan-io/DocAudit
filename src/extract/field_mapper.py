"""OCR output → schema fields. DeepSeek's primary job.

The mapper receives BOTH the raw text and a compact layout hint built
from PaddleOCR's regions. The hint is what makes column-order recovery
possible — receipt column order varies by store.

The mapper NEVER corrects arithmetic. A "fixed" total would erase the
error Check 1 exists to find.
"""

import re

from src.shared import deepseek_client
from src.shared.models import LineItem, Receipt
from src.extract import ocr_engine

ANCHORS = {
    "receipt_total": ["TOTAL", "JUMLAH", "AMOUNT DUE", "GRAND TOTAL"],
    "tax":           ["TAX", "SST", "GST", "CUKAI"],
    "cash":          ["CASH", "TUNAI", "PAID", "TENDER"],
    "change":        ["CHANGE", "BAKI"],
    "discount":      ["DISCOUNT", "DISKAUN", "POTONGAN"],
}


def _anchor_hints(text: str) -> dict:
    found, upper = {}, text.upper()
    for field, words in ANCHORS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", upper):
                found[field] = True
                break
    return found


def _to_items(rows: list, regions: list) -> list:
    """Row dicts → LineItem objects, carrying provenance."""
    items = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue                        # drop unusable rows
        items.append(LineItem(
            name=name,
            qty=float(r.get("qty") or 0),
            unit_price=float(r.get("unit_price") or 0),
            line_total=float(r.get("line_total") or 0),
            mismatch=bool(r.get("mismatch", False)),
            raw_text=r.get("raw") or str(r),
            region=ocr_engine.region_for_text(regions, name),
            ocr_confidence=_confidence_for(name, regions),
        ))
    return items


def _confidence_for(name: str, regions: list) -> float:
    """Carry PaddleOCR's per-box confidence onto the line item."""
    n = name.strip().lower()
    for r in regions:
        if n and n in (r.get("text") or "").lower():
            return float(r.get("confidence", 1.0))
    return 1.0


def map_receipt(ocr_result: dict, receipt_id: str, image_path: str,
                use_cache: bool = True) -> Receipt:
    """Full mapping pass: scalars + item table, both via DeepSeek."""
    clean_text = ocr_result["text"]
    regions = ocr_result["regions"]

    scalars = deepseek_client.map_scalars(clean_text, regions,
                                          use_cache=use_cache)
    rows = deepseek_client.map_items(clean_text, regions,
                                     use_cache=use_cache)

    return Receipt(
        vendor=scalars.get("vendor"),
        date=scalars.get("date"),
        currency="MYR",                          # enforced, not inferred
        items=_to_items(rows, regions),
        receipt_total=scalars.get("receipt_total"),
        tax=scalars.get("tax"),
        receipt_id=receipt_id,
        image_path=image_path,
        image_size=tuple(ocr_result["image_size"]),
        regions=regions,
        provenance={"_anchor_hints": _anchor_hints(clean_text),
                    "_reader": ocr_result.get("reader", "paddleocr")},
    )


def validate_currency(receipt: Receipt) -> bool:
    """Scope guard: one region, one currency. Never pool non-MYR."""
    return receipt.currency == "MYR"
