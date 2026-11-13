"""Printed tax lines and 'NAME 2 x 5.00' quantities (fix 13/11).

Synthetic regions in PaddleOCR's shape. Before the fix a clean receipt
with SST was flagged (tax ignored) and 'BREAD 2 x 5.00' was read as
qty 1, flagging a correct line.
"""

from src.detect.detect import Detector
from src.extract.hybrid_extract import structure_regions
from src.extract.receipt_parser import find_totals, parse_items


def _regions(lines):
    """lines: [(left_text, right_text_or_None)] → one row per entry."""
    regs, y = [], 20
    for left, right in lines:
        if left:
            regs.append({"text": left, "bbox_px": [20, y, 300, y + 24],
                         "confidence": 0.99})
        if right:
            regs.append({"text": right, "bbox_px": [480, y, 580, y + 24],
                         "confidence": 0.99})
        y += 40
    return regs


HEAD = [("NORTH MARKET SDN BHD", None), ("12/11/2026 09:14", None),
        ("BREAD 2 x 5.00", "10.00"), ("COFFEE 1 x 12.00", "12.00")]


def _flags(res, level=None):
    return [a for a in res["anomalies"] if a["anomaly_type"] == "incorrect_total"
            and (level is None or a["evidence"].get("line_index") is None)]


def test_trailing_qty_is_read():
    items = parse_items(_regions(HEAD))
    bread = items[0]
    assert bread["name"] == "BREAD"
    assert (bread["qty"], bread["unit_price"], bread["line_total"]) == (2.0, 5.0, 10.0)
    assert "2 x" in bread["raw_text"]            # observed string kept


def test_leading_qty_still_read():
    items = parse_items(_regions([("NORTH MARKET", None), ("2 x MILK", "13.00")]))
    assert (items[0]["name"], items[0]["qty"]) == ("MILK", 2.0)


def test_size_digits_stay_in_name():
    items = parse_items(_regions([("NORTH MARKET", None),
                                  ("HD40 1L", "6.50")]))
    assert items[0]["name"] == "HD40 1L" and items[0]["qty"] == 1.0


def test_clean_receipt_with_sst_not_flagged():
    regs = _regions(HEAD + [("SUBTOTAL", "22.00"), ("SST 6%", "3.00"),
                            ("TOTAL", "25.00"), ("CASH", "30.00"),
                            ("CHANGE", "5.00")])
    r = structure_regions(regs)
    assert r.receipt_total == 25.0                # not CASH
    assert r.tax == 3.0
    assert r.provenance["_tax"]["mode"] == "exclusive"
    assert _flags(Detector().run(r)) == []


def test_wrong_total_with_tax_explains_the_right_gap():
    regs = _regions(HEAD + [("SUBTOTAL", "22.00"), ("SST 6%", "3.00"),
                            ("TOTAL", "27.00")])
    flags = _flags(Detector().run(structure_regions(regs)))
    assert len(flags) == 1
    reason = flags[0]["reason"]
    assert "RM 2.00 higher" in reason and "RM 25.00" in reason
    assert "tax RM 3.00" in reason


def test_inclusive_tax_not_added():
    # subtotal already equals total: the GST line only states the tax inside
    regs = _regions(HEAD + [("SUBTOTAL", "22.00"), ("GST 6%", "1.25"),
                            ("TOTAL", "22.00")])
    r = structure_regions(regs)
    assert r.tax is None
    assert r.provenance["_tax"]["mode"] == "inclusive"
    assert r.provenance["_tax"]["printed"] == 1.25   # observed value kept
    assert _flags(Detector().run(r)) == []


def test_gst_summary_below_total_ignored():
    regs = _regions(HEAD + [("TOTAL", "22.00"), ("GST SUMMARY", None),
                            ("SR @ 6% GST", "1.25")])
    t = find_totals(regs)
    assert t["total"] == 22.0 and t["tax"] is None


def test_tax_label_rows_that_are_not_amounts_ignored():
    regs = _regions([("TAX INVOICE", None), ("GST REG NO 0012", "345.00")]
                    + HEAD + [("TOTAL INCL GST", "22.00"),
                              ("ROUNDING", "0.00"), ("TOTAL", "22.00")])
    t = find_totals(regs)
    assert t["total"] == 22.0 and t["tax"] is None


def test_same_tax_printed_twice_counted_once():
    regs = _regions(HEAD + [("SUBTOTAL", "22.00"), ("SST 6%", "3.00"),
                            ("TOTAL SST", "3.00"), ("TOTAL", "25.00")])
    t = find_totals(regs)
    assert t["tax"] == 3.0 and len(t["tax_rows"]) == 2


def test_gst_without_printed_subtotal_not_added():
    # SROIE train_105 shape: prices include GST, no SUBTOTAL row,
    # 'Total Excluding GST' + 'GST payable' printed above TOTAL
    regs = _regions([("NORTH MARKET", None), ("RICE", "18.00"),
                     ("TEA", "3.20"), ("Total Excluding GST", "20.00"),
                     ("GST payable 6%", "1.20"), ("TOTAL", "21.20")])
    r = structure_regions(regs)
    assert r.tax is None
    assert r.provenance["_tax"]["mode"] == "unknown"
    assert _flags(Detector().run(r)) == []


def test_service_charge_without_subtotal_added():
    regs = _regions([("NORTH MARKET", None), ("TIGER BUCKET", "55.00"),
                     ("SERVICE CHARGE 10%", "5.50"), ("TOTAL", "60.50")])
    r = structure_regions(regs)
    assert r.tax == 5.5
    assert r.provenance["_tax"]["mode"] == "service_only"
    assert _flags(Detector().run(r)) == []


# ── real PaddleOCR 2.10 output on the NORTH MARKET test receipts (13/11),
#    images in test_receipts_moi/ of the handoff folder
import json as _json
from pathlib import Path as _Path

_NORTH = _json.loads((_Path(__file__).parent / "fixtures" /
                      "paddle_north_regions.json").read_text(encoding="utf-8"))


def _real(name):
    r = structure_regions(_NORTH[name], receipt_id=name)
    return r, Detector().run(r)


def test_real_north_clean_no_flags():
    r, res = _real("north_clean.png")
    assert r.receipt_total == 25.0 and r.tax == 3.0
    assert [(i.name, i.qty) for i in r.items] == [("BREAD", 2.0), ("COFFEE", 1.0)]
    assert res["anomalies"] == []


def test_real_north_wrong_total_gap_is_two():
    r, res = _real("north_wrong_total.png")
    flags = _flags(res)
    assert len(flags) == 1 and "RM 2.00 higher" in flags[0]["reason"]


def test_real_north_duplicate():
    _, res = _real("north_dup.png")
    dup = [a for a in res["anomalies"] if a["anomaly_type"] == "duplicate"]
    assert len(dup) == 1 and "'MILK 1L'" in dup[0]["reason"]
