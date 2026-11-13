"""Locks the image-path parser on the three demo receipts.

Regions below mimic PaddleOCR output (one region per text segment, pixel
boxes) for demo/test_image_*.  Coordinates were taken from a real OCR
pass over those images; texts are what is printed on the receipt.
"""

import json
from pathlib import Path

from src.extract.receipt_parser import (parse_items, find_totals, to_money,
                                        group_rows)
from src.extract.hybrid_extract import structure_regions
from src.detect.detect import Detector

ROOT = Path(__file__).resolve().parents[1]


def R(x1, y1, x2, y2, text, conf=0.97):
    return {"label": "text", "bbox_px": [x1, y1, x2, y2], "text": text,
            "confidence": conf}


# demo/test_image_clean.png — OJC MARKETING (300 px wide)
OJC = [
    R(44, 9, 205, 18, "OJC MARKETING SDN BHD"),
    R(72, 24, 177, 33, "ROC NO: 538358-H"),
    R(100, 127, 149, 137, "Cash Bill"),
    R(20, 150, 70, 160, "Invoice No"), R(92, 150, 190, 160, ": PEGIV-1030531"),
    R(20, 164, 43, 173, "Date"), R(92, 164, 215, 174, ": 02/01/2019 2:47:14 PM"),
    R(20, 178, 144, 187, "Cashier : RHYS TAN"),
    R(20, 193, 122, 202, "Sales Person : FATIN"),
    R(16, 221, 70, 232, "Description"), R(110, 221, 128, 232, "Qty"),
    R(140, 221, 165, 232, "Price"), R(180, 221, 215, 232, "Amount"),
    R(16, 235, 62, 246, "000000111"), R(117, 235, 123, 246, "1"),
    R(138, 235, 165, 246, "170.00"), R(185, 235, 215, 246, "170.00"),
    R(18, 249, 81, 258, "KINGS SAFETY"), R(118, 249, 138, 258, "P004"),
    R(18, 261, 89, 270, "SHOES KWD 805"),
    R(18, 280, 230, 292, "Qty: 1 Total Item Discount:"), R(200, 280, 215, 292, "0.00"),
    R(84, 296, 150, 306, "Total Amount:"), R(185, 296, 215, 306, "170.00"),
    R(109, 329, 150, 340, "TOTAL:"), R(175, 329, 215, 340, "170.00"),
    R(88, 347, 137, 355, "VISA CARD"), R(185, 347, 215, 355, "170.00"),
]

# demo/test_image_incorrect_total.jpg — SOON HUAT (600 px), Total Qty row altered to 377.00
_SH_ROWS = [
    ("1072", "1", "80.00", "80.00", "80.00",
     ["REPAIR ENGINE POWER SPRAYER (1UNIT)", "workmanship & service"]),
    ("70549", "1", "160.00", "160.00", "160.00", ["GIANT 606 OVERFLOW ASSY"]),
    ("1071", "1", "17.00", "17.00", "17.00", ["ENGINE OIL"]),
    ("70791", "1", "10.00", "10.00", "10.00", ["GREASE FOR TOOLS 40ML (AKODA)"]),
    ("70637", "1", "6.00", "6.00", "6.00", ["EY20 PLUG CHAMPION"]),
    ("1643", "1", "8.00", "8.00", "8.00", ["STARTER TALI"]),
    ("70197", "1", "10.00", "10.00", "10.00", ["EY20 STARTER HANDLE"]),
    ("70561", "2", "18.00", "18.00", "36.00", ["HD40 1L COTIN"]),
]


def _soon_huat():
    regs = [
        R(21, 14, 571, 40, "SOON HUAT MACHINERY ENTERPRISE"),
        R(214, 328, 380, 350, "CASH SALES"),
        R(6, 372, 90, 392, "Doc No."), R(156, 369, 316, 391, ": CS00004040"),
        R(399, 368, 591, 390, "Date: 11/01/2019"),
        R(5, 405, 88, 425, "Cashier"), R(156, 405, 241, 426, ": USER"),
        R(6, 504, 46, 525, "Item"), R(177, 504, 211, 525, "Qty"),
        R(247, 504, 316, 525, "S/Price"), R(342, 504, 410, 525, "S/Price"),
        R(470, 504, 527, 525, "Amount"), R(553, 504, 588, 525, "Tax"),
    ]
    y = 536
    for code, qty, p1, p2, amt, desc in _SH_ROWS:
        regs += [R(8, y, 60, y + 18, code), R(190, y, 198, y + 18, qty),
                 R(256, y, 310, y + 18, p1), R(350, y, 404, y + 18, p2),
                 R(477, y, 530, y + 18, amt)]
        for d in desc:
            y += 34
            regs.append(R(8, y, 8 + 11 * len(d), y + 18, d))
        y += 36
    regs += [
        R(38, y, 160, y + 25, "Total Qty:"), R(190, y, 200, y + 25, "9"),
        R(470, y, 535, y + 25, "377.00"),
        R(272, y + 60, 411, y + 82, "Total Sales :"), R(463, y + 60, 535, y + 82, "327.00"),
        R(298, y + 102, 389, y + 122, "Discount :"), R(489, y + 102, 535, y + 122, "0.00"),
        R(339, y + 144, 390, y + 164, "Total :"), R(490, y + 144, 535, y + 164, "0.00"),
        R(292, y + 186, 412, y + 206, "Rounding :"), R(490, y + 186, 535, y + 206, "0.00"),
        R(263, y + 228, 411, y + 250, "Total Sales :"), R(463, y + 228, 535, y + 250, "327.00"),
        R(390, y + 270, 450, y + 292, "CASH :"), R(463, y + 270, 535, y + 292, "327.00"),
    ]
    return regs


# demo/test_image_duplicate_item.jpg — ABC HO TRADING (770 px)
ABC = [
    R(200, 14, 500, 45, "ABC HO TRADING"),
    R(16, 280, 170, 310, "Cash Bill :"), R(185, 280, 330, 310, "01-143008"),
    R(16, 320, 110, 350, "Date"), R(185, 320, 490, 350, "09/01/2019 8:01:11 PM"),
    R(18, 362, 112, 388, "Cashier"), R(185, 363, 215, 387, "01"),
    R(38, 400, 206, 435, "Description"), R(347, 404, 398, 433, "Qty"),
    R(449, 405, 520, 430, "Price"), R(550, 405, 667, 430, "Amount"),
    R(450, 440, 500, 465, "RM"), R(560, 440, 610, 465, "RM"),
    R(16, 486, 105, 511, "Plastic"), R(381, 488, 397, 513, "2"),
    R(433, 485, 503, 510, "15.50"), R(550, 489, 632, 515, "31.00"),
    R(16, 528, 105, 553, "Plastic"), R(381, 530, 397, 555, "2"),
    R(433, 527, 503, 552, "15.50"), R(550, 529, 632, 555, "31.00"),
    R(337, 572, 411, 600, "Total :"), R(608, 576, 677, 600, "31.00"),
    R(223, 615, 412, 643, "Total Amount:"), R(590, 610, 684, 648, "31.00"),
    R(207, 658, 395, 682, "Cash Received :"), R(594, 659, 677, 684, "101.00"),
    R(291, 698, 410, 728, "Change :"), R(608, 699, 677, 724, "70.00"),
]


def test_money_tokens():
    assert to_money("170.00") == 170.0
    assert to_money("17,00") == 17.0
    assert to_money("RM 4.50") == 4.5
    assert to_money("1072") is None
    assert to_money("09:44:00") is None


def test_row_grouping_scales_with_resolution():
    rows = group_rows(OJC)
    assert any("TOTAL:" in " ".join(r["text"] for r in row) for row in rows)


def test_description_below_code_row():
    items = parse_items(OJC)
    assert len(items) == 1
    it = items[0]
    assert it["name"] == "KINGS SAFETY SHOES KWD 805"     # not '000000111', no 'P004'
    assert (it["qty"], it["unit_price"], it["line_total"]) == (1.0, 170.0, 170.0)


def test_total_label_and_value_in_separate_regions():
    t = find_totals(OJC)
    assert t["total"] == 170.0
    assert t["total_region"]


def test_soon_huat_items_and_printed_subtotal():
    regs = _soon_huat()
    items = parse_items(regs)
    assert [i["name"] for i in items][:2] == [
        "REPAIR ENGINE POWER SPRAYER (1UNIT) workmanship & service",
        "GIANT 606 OVERFLOW ASSY"]
    assert len(items) == 8
    assert round(sum(i["line_total"] for i in items), 2) == 327.0
    assert items[-1]["qty"] == 2.0 and items[-1]["unit_price"] == 18.0
    t = find_totals(regs)
    assert t["total"] == 327.0          # 'Total : 0.00' is ignored
    assert t["subtotal"] == 377.0       # the altered 'Total Qty' amount


def test_soon_huat_flags_altered_subtotal():
    r = structure_regions(_soon_huat(), receipt_id="soon_huat")
    res = Detector().run(r)
    reasons = [a["reason"] for a in res["anomalies"]]
    assert any("377.00" in x and "327.00" in x for x in reasons), reasons
    assert all(a["anomaly_type"] == "incorrect_total" for a in res["anomalies"])


def test_abc_duplicate_rows_detected():
    r = structure_regions(ABC, receipt_id="abc")
    assert [i.name for i in r.items] == ["Plastic", "Plastic"]
    assert r.receipt_total == 31.0
    res = Detector().run(r)
    types = [a["anomaly_type"] for a in res["anomalies"]]
    assert "duplicate" in types


def test_clean_receipt_has_no_flags():
    r = structure_regions(OJC, receipt_id="ojc")
    assert Detector().run(r)["anomalies"] == []


def test_demo_unusual_price_needs_demo_pool():
    """demo_unusual_price.json only fires with the demo reference prices."""
    from src.shared.models import Receipt, LineItem
    data = json.loads((ROOT / "demo/demo_unusual_price.json").read_text())
    pool = json.loads((ROOT / "demo/demo_price_pool.json").read_text())
    r = Receipt(vendor=data["vendor"], date=data["date"],
                receipt_total=data["receipt_total"], tax=data["tax"],
                items=[LineItem(i["name"], i["qty"], i["unit_price"],
                                i["line_total"]) for i in data["items"]])
    res = Detector(price_pool=pool).run(r)
    flagged = {a["evidence"]["canonical"] for a in res["anomalies"]
               if a["anomaly_type"] == "unusual_price"}
    assert flagged == {"MILK 1L", "BREAD WHITE"}
