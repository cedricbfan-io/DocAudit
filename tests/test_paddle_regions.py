"""Regression tests on REAL PaddleOCR output for the three demo images.

tests/fixtures/paddle_demo_regions.json was recorded by running
PaddleOCR 2.10.0 on demo/test_image_* (11/11). Unlike the synthetic
fixtures in test_receipt_parser.py it keeps the OCR's own quirks:
merged columns ('1170.00'), misreads ('Plastie', 'KWD S05') and
labels split from their amounts.
"""

import json
from pathlib import Path

import pytest

from src.detect.detect import Detector
from src.extract.hybrid_extract import structure_regions
from src.extract.receipt_parser import _split_merged_qty, find_totals

FIX = json.loads((Path(__file__).parent / "fixtures" /
                  "paddle_demo_regions.json").read_text(encoding="utf-8"))


def _run(name):
    r = structure_regions(FIX[name], receipt_id=name)
    return r, Detector().run(r)


def test_clean_merged_qty_price_token_is_split():
    r, res = _run("test_image_clean.png")
    assert len(r.items) == 1
    it = r.items[0]
    assert it.name.startswith("KINGS SAFETY SHOES KWD")
    assert (it.qty, it.unit_price, it.line_total) == (1.0, 170.0, 170.0)
    assert "1170.00" in it.raw_text                 # observed string kept
    assert r.provenance["_derived_fields"][0]["row"] == 0
    assert res["anomalies"] == []


def test_clean_total_is_the_bottom_total_row():
    r, _ = _run("test_image_clean.png")
    assert r.receipt_total == 170.0
    # 'TOTAL:' row (y≈312-338), not 'Total Amount:' (y≈280-308)
    y1 = int(r.provenance["_total_region"].split(",")[1])
    assert y1 >= 310


def test_duplicate_found_despite_ocr_misread():
    r, res = _run("test_image_duplicate_item.jpg")
    assert [i.name for i in r.items] == ["Plastic", "Plastie"]
    dup = [a for a in res["anomalies"] if a["anomaly_type"] == "duplicate"]
    assert len(dup) == 1
    assert dup[0]["evidence"]["line_indexes"] == [0, 1]
    assert r.receipt_total == 31.0                  # 'Total Amount:' row


def test_incorrect_total_flags_total_qty_row():
    r, res = _run("test_image_incorrect_total.jpg")
    assert len(r.items) == 8
    assert round(r.subtotal(), 2) == 327.0
    assert r.receipt_total == 327.0                 # bottom 'Total Sales'
    assert r.printed_subtotal == 377.0              # altered 'Total Qty'
    [a] = res["anomalies"]
    assert a["anomaly_type"] == "incorrect_total"
    assert a["evidence"]["region"] == r.provenance["_subtotal_region"]


@pytest.mark.parametrize("value,amount,expected", [
    (1170.0, 170.0, (1.0, 170.0)),
    (218.0, 36.0, (2.0, 18.0)),
    (17.0, 17.0, None),            # no consistent split → leave alone
    (1170.0, 999.0, None),
])
def test_split_merged_qty(value, amount, expected):
    assert _split_merged_qty(value, amount) == expected


def R(x1, y1, x2, y2, text):
    return {"bbox_px": [x1, y1, x2, y2], "text": text, "confidence": 0.99}


def test_cash_and_change_are_never_the_total():
    regs = [R(10, 10, 80, 30, "TOTAL"), R(200, 10, 260, 30, "25.00"),
            R(10, 40, 120, 60, "TOTAL TENDERED"), R(200, 40, 260, 60, "30.00"),
            R(10, 70, 80, 90, "CHANGE"), R(200, 70, 260, 90, "5.00")]
    assert find_totals(regs)["total"] == 25.0


def test_duplicate_image_62_vs_31_is_not_silently_skipped():
    """Case (a) on REAL OCR: Plastic 31.00 + Plastie 31.00 = 62.00 vs
    TOTAL 31.00. The coverage guard (ratio 2.0) used to end the check
    with no message; now the gap is explained by the repeated line."""
    r, res = _run("test_image_duplicate_item.jpg")
    tc = res["total_check"]
    assert tc["status"] == "mismatch_explained_by_duplicate"
    assert (tc["items_sum"], tc["observed"], tc["difference"]) == (62.0, 31.0, -31.0)
    [dup] = res["anomalies"]
    assert dup["anomaly_type"] == "duplicate"
    assert "RM 62.00" in dup["reason"] and "'Plastic'" in dup["reason"]


def test_clean_and_incorrect_total_images_are_checked():
    for name in ("test_image_clean.png", "test_image_incorrect_total.jpg"):
        _, res = _run(name)
        assert res["total_check"]["status"] == "checked"
