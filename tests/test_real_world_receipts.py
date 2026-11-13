"""Receipts outside SROIE (13/11): formats, labels and layouts that made
the app miss or misread the TOTAL when the team tried their own photos."""

import pytest

from src.detect.detect import Detector
from src.extract.hybrid_extract import _detect_currency, structure_regions
from src.extract.receipt_parser import find_totals, parse_items, to_money


def _regions(lines, conf=0.99):
    """[(left_text, right_text)] → PaddleOCR-shaped regions, one row each."""
    regs, y = [], 20
    for left, right in lines:
        if left:
            regs.append({"text": left, "bbox_px": [20, y, 300, y + 24],
                         "confidence": conf})
        if right:
            regs.append({"text": right, "bbox_px": [480, y, 580, y + 24],
                         "confidence": conf})
        y += 40
    return regs


H = [("CORNER MART", None), ("12/11/2026 18:02", None)]


@pytest.mark.parametrize("token,value", [
    ("17.00", 17.0), ("RM17,00", 17.0), ("$27.35", 27.35), ("¥17.4", 17.4),
    ("1,234.50", 1234.5), ("RM1,234.50", 1234.5), ("-RM0.02", -0.02),
    ("27.35RM", 27.35), ("RH60.30", 60.3), ("1,50", 1.5),
    ("17.0", None), ("12", None), ("3.5", None), ("2026", None),
])
def test_money_tokens(token, value):
    assert to_money(token) == value


def test_gst_summary_total_below_payment_is_ignored():
    # SROIE train_008 shape: the summary's 'Total: 106.10 6.37' used to win
    regs = _regions(H + [("PAINT BRUSH", "11.00"), ("ROLLER", "95.10"),
                         ("Total GST (RM", "6.37"), ("Total (RM", "112.45"),
                         ("CASH", "112.45"), ("GST SUMMARY", None),
                         ("SR 6 106.10", "6.37"), ("Total: 106.10", "6.37")])
    assert find_totals(regs)["total"] == 112.45


def test_rounded_total_without_label():
    # SROIE train_001: 'TOTAL AMT 60.31' / 'ROUNDING ADJ -0.01' / 'RH 60.30'
    regs = _regions(H + [("ITEM A", "60.31"), ("TOTAL AMT RM", "60.31"),
                         ("ROUNDING ADJ", "-0.01"), ("RH", "60.30"),
                         ("CASH. RM", "70.30"), ("CHANGE. RM", "10.00")])
    assert find_totals(regs)["total"] == 60.30


@pytest.mark.parametrize("label", [
    "Total Sales (lnclusive of GsT)", "Tota1 RM Including GST 6%",
    "Item2 Total with GST@6%", "TOTAL ROUNDED", "AMOUNT DUE", "Balance Due",
    "Grand Total:", "Amt Due", "合计", "应付金额：",
])
def test_total_label_variants(label):
    regs = _regions(H + [("COFFEE", "77.20"), (label, "77.20"), ("CASH", "100.00")])
    assert find_totals(regs)["total"] == 77.20


def test_label_and_amount_on_separate_lines():
    regs = _regions(H + [("APPLE", "1.00"), ("RICE 5KG", "26.35"), ("TOTAL", None),
                         (None, "RM 27.35"), ("CASH", "RM 30.00")])
    assert find_totals(regs)["total"] == 27.35


def test_total_qty_and_savings_and_payment_are_not_the_total():
    regs = _regions(H + [("PEN", "2.40"), ("*Total Qty4.00", "7.80"),
                         ("TOTAL", "7.80"), ("TOTAL SAVINGS", "4.20"),
                         ("VISA ****4021", "7.80")])
    t = find_totals(regs)
    assert t["total"] == 7.80


def test_no_total_label_is_found_by_arithmetic_but_needs_review():
    # El Pollo Loco slip the team photographed: no 'TOTAL' line at all
    regs = _regions(H + [("1 Burrito Cmb", "8.28"), ("2 Bur-BRC @ 1.89", "3.78"),
                         ("Food", "12.06"), ("Tax", "1.06"),
                         ("Payment", "13.12"), ("Change Due", "0.13")])
    r = structure_regions(regs)
    assert r.receipt_total == 13.12
    assert r.provenance["_total_inferred"]["row"].startswith("Payment")
    res = Detector().run(r)
    assert res["total_check"]["status"] == "inferred"
    assert "No TOTAL label" in res["total_check"]["reason"]
    assert [a for a in res["anomalies"] if a["anomaly_type"] == "incorrect_total"] == []


def test_inferred_total_never_uses_an_item_row():
    regs = _regions(H + [("ONLY ITEM", "9.90"), ("THANK YOU", None)])
    r = structure_regions(regs)
    assert r.receipt_total is None


@pytest.mark.parametrize("left,right,name,qty,unit", [
    ("2 APPLE", "1.00", "APPLE", 2.0, 0.5),          # Green Supermarket
    ("1PEAR", "0.75", "PEAR", 1.0, 0.75),             # OCR glued the qty
    ("2Bur-BRC @1.89", "3.78", "Bur-BRC", 2.0, 1.89),
    ("2 Taco @ 1.89", "3.78", "Taco", 2.0, 1.89),
    ("鸡肉云吞 x1", "¥17.4", "鸡肉云吞", 1.0, 17.4),
    ("BREAD 2 @ 5.00", "10.00", "BREAD", 2.0, 5.0),
])
def test_quantity_layouts(left, right, name, qty, unit):
    [it] = parse_items(_regions([("SHOP", None), (left, right), ("TOTAL", right)]))
    assert (it["name"], it["qty"]) == (name, qty)
    assert it["unit_price"] == pytest.approx(unit)


@pytest.mark.parametrize("left", ["3 IN 1 COFFEE", "7UP 1.5L", "100PLUS"])
def test_numbers_that_belong_to_the_name(left):
    [it] = parse_items(_regions([("SHOP", None), (left, "4.50"), ("TOTAL", "4.50")]))
    assert it["qty"] == 1.0 and it["name"].replace(" ", "")[:3] == left.replace(" ", "")[:3]


@pytest.mark.parametrize("lines,code", [
    (["TOTAL RM 12.00"], "MYR"), (["TOTAL $23.27"], "USD"),
    (["合计 ¥45.00"], "CNY"), (["北京理工大学良乡校区店 拿铁"], "CNY"),
    (["TOTAL 9.70"], "MYR"),
])
def test_currency_detection(lines, code):
    assert _detect_currency(lines) == code


def test_non_myr_receipt_is_checked_but_not_priced():
    regs = _regions([("便利店", None), ("MILK 1L", "¥27.00"), ("合计", "¥27.00")])
    r = structure_regions(regs)
    assert r.currency == "CNY" and r.receipt_total == 27.0
    pool = {"MILK 1L": [3.5, 3.6, 3.4, 3.5]}            # ringgit prices
    res = Detector(price_pool=pool).run(r)
    assert not any(a["anomaly_type"] == "unusual_price" for a in res["anomalies"])


def test_messages_use_the_receipt_currency():
    regs = _regions(H + [("COFFEE", "$8.00"), ("BAGEL", "$10.00"), ("TOTAL", "$20.00")])
    r = structure_regions(regs)
    res = Detector().run(r)
    [a] = [a for a in res["anomalies"] if a["anomaly_type"] == "incorrect_total"]
    assert "$ 20.00" in a["reason"] and "RM" not in a["reason"]


def test_image_formats(tmp_path):
    from PIL import Image
    from src.extract.preprocessing import UnreadableImage, load_image
    im = Image.new("RGB", (300, 4000), "white")
    for ext in ("png", "jpg", "webp"):
        p = tmp_path / f"r.{ext}"
        im.save(p)
        arr = load_image(str(p))
        assert max(arr.shape[:2]) <= 2400            # big photos scaled down
    try:
        im.save(tmp_path / "r.avif")
        assert load_image(str(tmp_path / "r.avif")).shape[2] == 3
    except (KeyError, OSError):
        pytest.skip("this Pillow build has no AVIF writer")
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(UnreadableImage, match="Cannot open this file"):
        load_image(str(bad))


def test_rounded_unit_price_is_not_a_line_error():
    # SROIE train_014: 17 × 0.897 prints as '17 × 0.90 = 15.25'
    from src.shared.arithmetic import line_ok
    assert line_ok(17, 0.90, 15.25)
    assert line_ok(30, 0.13, 3.80)
    assert not line_ok(2, 3.50, 9.50)                 # a real line error stays


def test_trailing_zero_column_is_not_the_amount():
    # SROIE train_065: '... 35.00 0.00' (discount column)
    [it] = parse_items(_regions([("SHOP", None), ("Fish Head (S)", "35.00 0.00"),
                                 ("TOTAL", "35.00")]))
    assert it["line_total"] == 35.0


def test_code_read_as_quantity_is_dropped():
    # SROIE train_034: '801 6.00 6.00' — 801 is a code, the line is 1 × 6.00
    regs = [{"text": "COTTON GLOVE", "bbox_px": [20, 60, 200, 84], "confidence": .99},
            {"text": "801", "bbox_px": [300, 60, 340, 84], "confidence": .99},
            {"text": "6.00", "bbox_px": [400, 60, 450, 84], "confidence": .99},
            {"text": "6.00", "bbox_px": [500, 60, 560, 84], "confidence": .99}]
    [it] = parse_items([{"text": "SHOP", "bbox_px": [20, 20, 200, 44],
                         "confidence": .99}] + regs)
    assert (it["qty"], it["unit_price"], it["line_total"]) == (1.0, 6.0, 6.0)
    assert "code" in it["derived"]
