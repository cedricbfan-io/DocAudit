"""Render the 20 'varied' receipts used by eval/real_world.py.

    python -m eval.make_varied            # writes eval/varied/*.png + expected.json

Synthetic, so the true total is known. Each one targets a layout seen on
real receipts outside SROIE: '$' / '¥' amounts, 'AMOUNT DUE' / 'Balance
Due' / 'Grand Total' / '合计' / '应付金额', the amount on the line under
the label, a GST summary under the total, rounding, 'incl. GST', a
leading quantity, quantity on the line under the name, no TOTAL label at
all, savings and payment lines after the total, a tilted blurry photo,
and a slip with no total. Fonts: macOS system fonts (falls back to
Pillow's default font elsewhere, so images differ across machines —
the committed PNGs are the reference set).
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "varied"
FONTS = {
    "courier": "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "andale": "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "arial": "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "menlo": "/System/Library/Fonts/Menlo.ttc",
    "cjk": "/System/Library/Fonts/STHeiti Medium.ttc",
}


def _font(name, size):
    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default(size)


def render(path, lines, font="courier", size=24, width=560, rot=0.0, blur=0.0):
    f = _font(font, size)
    lh = int(size * 1.55)
    im = Image.new("RGB", (width, 60 + lh * len(lines)), (250, 248, 242))
    d = ImageDraw.Draw(im)
    y = 30
    for left, right in lines:
        if left:
            d.text((26, y), left, font=f, fill=(20, 20, 20))
        if right:
            d.text((width - 26 - d.textlength(right, font=f), y), right,
                   font=f, fill=(20, 20, 20))
        y += lh
    if rot:
        im = im.rotate(rot, expand=True, fillcolor=(235, 235, 235))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    im.save(path)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    expected = {}

    def rec(name, total, lines, **kw):
        expected[name + ".png"] = total
        render(OUT / f"{name}.png", lines, **kw)

    H = [("CORNER MART", None), ("12/11/2026 18:02", None), ("", None)]
    rec("v01_us_tax", 23.27, H + [("MILK 1GAL", "$4.29"), ("BREAD", "$3.49"),
        ("EGGS DOZEN", "$5.99"), ("COFFEE", "$8.00"), ("SUBTOTAL", "$21.77"),
        ("TAX 6.9%", "$1.50"), ("TOTAL", "$23.27"), ("VISA ****4021", "$23.27")])
    rec("v02_amount_due", 41.50, H + [("PIZZA LARGE", "28.00"), ("SODA 2L", "4.50"),
        ("DELIVERY", "9.00"), ("AMOUNT DUE", "41.50"), ("CASH", "50.00"),
        ("CHANGE", "8.50")], font="andale")
    rec("v03_balance_due", 18.90, H + [("Latte", "5.40"), ("Croissant", "3.50"),
        ("Sandwich", "10.00"), ("Balance Due", "18.90")], font="arial", size=26)
    rec("v04_split_label", 27.35, H + [("APPLE", "1.00"), ("BANANA", "1.50"),
        ("RICE 5KG", "24.85"), ("TOTAL", None), (None, "RM 27.35"),
        ("CASH", "RM 30.00"), ("CHANGE", "RM 2.65")])
    rec("v05_thousands", 1234.50, H + [("LAPTOP STAND", "89.50"),
        ("MONITOR 27IN", "1,145.00"), ("TOTAL", "1,234.50")], font="menlo", size=22)
    rec("v06_grand_total", 64.80, H + [("NASI LEMAK", "12.00"), ("TEH TARIK", "4.80"),
        ("CHICKEN RICE", "48.00"), ("Grand Total:", "64.80"), ("Paid", "70.00")])
    rec("v07_gst_summary", 112.45, H + [("PAINT BRUSH", "11.00"), ("ROLLER", "95.10"),
        ("Total GST", "6.37"), ("Rounding", "-0.02"), ("Total (RM)", "112.45"),
        ("CASH", "112.45"), ("GST SUMMARY", None), ("SR 6%  106.10", "6.37"),
        ("Total:  106.10", "6.37")])
    rec("v08_rounding", 10.00, H + [("NOODLES", "6.53"), ("ICE TEA", "3.50"),
        ("TOTAL", "10.03"), ("ROUNDING ADJ", "-0.03"), ("TOTAL ROUNDED", "10.00"),
        ("CASH", "20.00"), ("CHANGE", "10.00")])
    rec("v09_inclusive", 45.80, H + [("SHAMPOO", "22.90"), ("SOAP 3PK", "22.90"),
        ("Total (incl. GST)", "45.80"), ("GST 6% included", "2.59")],
        font="arial", size=26)
    rec("v10_leading_qty", 9.70, [("FRESH FRUIT", None), ("", None),
        ("2 APPLE", "1.00"), ("3 BANANA", "1.50"), ("1 MANGO", "4.20"),
        ("2 KIWI", "3.00"), ("TOTAL", "9.70"), ("CASH", "10.00"), ("CHANGE", "0.30")])
    rec("v11_qty_below", 16.40, H + [("CHOC BAR", None), ("  2 x 2.50", "5.00"),
        ("ORANGE JUICE", None), ("  3 x 3.80", "11.40"), ("TOTAL", "16.40")])
    rec("v12_no_label", 13.12, H + [("1 Burrito", "8.28"), ("2 Taco @ 1.89", "3.78"),
        ("Food", "12.06"), ("Tax", "1.06"), ("Payment", "13.12"),
        ("Change Due", "0.13")])
    rec("v13_cn_heji", 45.00, [("便利店", None), ("2026-11-13 09:10", None),
        ("矿泉水 x2", "¥6.00"), ("面包", "¥12.00"), ("牛奶 x3", "¥27.00"),
        ("合计", "¥45.00"), ("现金", "¥50.00"), ("找零", "¥5.00")], font="cjk", size=26)
    rec("v14_cn_yingfu", 38.50, [("小吃店", None), ("2026-11-13 12:30", None),
        ("牛肉面", "¥22.00"), ("饺子", "¥16.50"), ("应付金额：", "¥38.50"),
        ("实付", "¥38.50")], font="cjk", size=26)
    rec("v15_my_cash", 52.30, H + [("BERAS 5KG", "28.90"), ("MINYAK 2KG", "13.40"),
        ("GULA 1KG", "10.00"), ("TOTAL RM", "52.30"), ("TUNAI", "100.00"),
        ("BAKI", "47.70")])
    rec("v16_total_items", 7.80, H + [("PEN BLUE", "2.40"), ("NOTEBOOK", "5.40"),
        ("TOTAL ITEMS: 2", None), ("TOTAL", "7.80")])
    rec("v17_savings", 30.60, H + [("DETERGENT", "18.90"), ("SPONGE", "11.70"),
        ("TOTAL", "30.60"), ("TOTAL SAVINGS", "4.20"), ("CARD", "30.60")])
    rec("v18_service", 33.65, H + [("FRIED RICE", "14.00"), ("LAKSA", "15.00"),
        ("SUB TOTAL", "29.00"), ("SERVICE CHARGE 10%", "2.90"), ("SST 6%", "1.74"),
        ("ROUNDING", "0.01"), ("NET TOTAL", "33.65")], font="andale")
    rec("v19_rotated", 22.00, H + [("BREAD 2 x 5.00", "10.00"),
        ("COFFEE 1 x 12.00", "12.00"), ("TOTAL", "22.00"), ("CASH", "30.00"),
        ("CHANGE", "8.00")], rot=4, blur=0.6)
    rec("v20_no_total", None, H + [("RECEIPT COPY", None), ("ORDER #4471", None),
        ("THANK YOU", None)])
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1,
                                                  ensure_ascii=False))
    return expected


if __name__ == "__main__":
    print(len(build()), "receipts written to", OUT)
