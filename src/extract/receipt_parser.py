"""Layout-aware receipt parser: OCR regions → header fields + line items.

Replaces the fixed-pixel column parser for the image pipeline. Works on
the region list PaddleOCR returns (one region per detected text line
segment, with a pixel bbox), and needs no LLM.

Why a rewrite was needed (demo images, 11/11):
  * Row grouping used a fixed 15 px tolerance and header columns used
    fixed pixel widths (x+50, x+80). Both break when image resolution
    changes (the demo images are 300 px and 600-770 px wide).
  * Many Malaysian receipts print a numeric row (code, qty, price,
    amount) and put the DESCRIPTION on the line(s) BELOW it. The old
    parser named the item after its product code ("000000111") and
    dropped the description row because it had no amount.
  * The total finder ran a regex over single regions, but PaddleOCR
    returns "TOTAL:" and "170.00" as two separate regions, so
    receipt_total was almost always None and the incorrect-total check
    could never fire on a photo.

Everything here is deterministic and keeps pixel evidence: each item
and each total carries the bbox of the row(s) it was read from.
"""

import re
from statistics import median

# ── token patterns ────────────────────────────────────────
_MONEY = re.compile(r"^(?:RM|MYR)?\s*-?\d{1,6}(?:[.,:]\d{2})$", re.IGNORECASE)
_INT = re.compile(r"^\d{1,3}$")
_CODE = re.compile(r"^[A-Z]{0,3}\d{3,}[A-Z]?$", re.IGNORECASE)
_DATE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")

# header row: a description-like label + at least one numeric column label
_HDR_NAME = re.compile(r"\b(DESC\w*|ITEMS?|PRODUCT|PARTICULARS?|PERKARA)\b", re.I)
_HDR_NUM = re.compile(r"\b(QTY|QUANTITY|PRICE|S/PRICE|U/PRICE|UNIT|AMOUNT|AMT|RM)\b", re.I)
_HDR_ANY = re.compile(r"(QTY|QUANTITY|PRICE|AMOUNT|\bAMT\b|\bTAX\b|DESC|\bITEM)", re.I)

# metadata rows that can appear above the item block
_META = re.compile(r"\b(DATE|TIME|CASHIER|SALES\s*PERSO\w*|SALESPERSON|INVOICE|"
                   r"DOC\s*NO|BILL|REF|TABLE|RECEIPT\s*NO|COUNTER)\b", re.I)

# rows that end the item block / are never items
_STOP = re.compile(r"\b(SUB\s*-?\s*TOTAL|TOTAL\w*|JUMLAH|ROUNDING|ROUND\s*AMT|"
                   r"DISCOUNT|DISKAUN|CASH|CHANGE|BAKI|TUNAI|TENDER\w*|"
                   r"BALANCE|VISA|MASTER\w*|DEBIT|CREDIT|GST|SST|"
                   r"TAX|SERVICE\s*CHARGE|THANK|GOODS\s*SOLD)\b", re.I)

# total labels, best first. Each entry: (pattern, kind)
_TOTAL_LABELS = [
    (re.compile(r"\b(GRAND\s*TOTAL|TOTAL\s*AMOUNT|NETT?\s*TOTAL|"
                r"TOTAL\s*SALES|AMOUNT\s*DUE|TOTAL\s*\(?RM\)?|JUMLAH)\b", re.I), "total"),
    (re.compile(r"\bTOTAL\w*", re.I), "total"),
]
_TRAILING_QTY = re.compile(r"^(.*\S)\s+(\d{1,3})\s*[xX*@]$")

# printed tax / service-charge rows. Rows that only MENTION tax (a GST
# summary table, 'Total incl. GST', 'GST Reg No', 'Tax Invoice') are not
# an amount added on top.
_TAX_LABEL = re.compile(r"\b(SST|GST|TAX|SERVICE\s*CH\w*|SVC\s*CHG|CUKAI)\b", re.I)
_SERVICE_LABEL = re.compile(r"\b(SERVICE\s*CH\w*|SVC\s*CHG)\b", re.I)
_TAX_NOT = re.compile(r"\b(INCL\w*|EXCL\w*|SUMMARY|REG\w*|INVOICE|ID|NO|"
                      r"CODE|ROUND\w*)\b", re.I)

_SUBTOTAL_LABEL = re.compile(r"\b(SUB\s*-?\s*TOTAL|TOTAL\s*QTY)\b", re.I)
_NOT_TOTAL = re.compile(r"\b(DISCOUNT|DISKAUN|ROUND\w*|TAX|GST|SST|"
                        r"ITEM\s*DISCOUNT|SAVINGS?|"
                        # money handed over / given back is never the total
                        r"CASH|CHANGE|TENDER\w*|PAID|RECEIVED|BAKI|TUNAI)\b",
                        re.I)


# ── helpers ───────────────────────────────────────────────
def to_money(text: str):
    """'RM 17,00' → 17.0; returns None if the token is not a money amount."""
    t = text.strip().replace(" ", "")
    if not _MONEY.match(t):
        return None
    t = re.sub(r"^(RM|MYR)", "", t, flags=re.I)
    t = t[:-3] + "." + t[-2:]           # ',' or '.' as the decimal mark
    try:
        return float(t)
    except ValueError:
        return None


def _union(bboxes):
    xs1, ys1, xs2, ys2 = zip(*bboxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def box_str(bbox) -> str:
    return ",".join(str(int(v)) for v in bbox)


def group_rows(regions):
    """Group regions into visual rows using each region's vertical centre.

    Tolerance scales with the median text height, so it works for
    300 px and 3000 px images alike.
    """
    regs = [r for r in regions if (r.get("text") or "").strip()]
    if not regs:
        return []
    heights = [r["bbox_px"][3] - r["bbox_px"][1] for r in regs]
    tol = max(4.0, 0.55 * median(heights))

    regs.sort(key=lambda r: (r["bbox_px"][1] + r["bbox_px"][3]) / 2)
    rows, cur, cur_c = [], [], None
    for r in regs:
        c = (r["bbox_px"][1] + r["bbox_px"][3]) / 2
        if cur and abs(c - cur_c) > tol:
            rows.append(sorted(cur, key=lambda q: q["bbox_px"][0]))
            cur = []
        if not cur:
            cur_c = c
        else:
            cur_c = (cur_c * len(cur) + c) / (len(cur) + 1)
        cur.append(r)
    if cur:
        rows.append(sorted(cur, key=lambda q: q["bbox_px"][0]))
    return rows


def _row_text(row):
    return " ".join(r["text"].strip() for r in row)


def _split_tokens(row):
    """Split a row into (text_parts, numeric_tokens).

    numeric token = (kind, value, region), kind 'money' or 'int'.
    Money-looking words ("17.00", "17,00") are always numeric. Plain
    integers count as numbers only when their region holds nothing but
    numbers — so "HD40 1L" or "MILO 3 IN 1" keep their digits in the name.
    Text parts are returned as region-like dicts with the numbers removed.
    """
    text_parts, nums = [], []
    for r in row:
        words = [w.strip(":=_;|.\u2014\"'°") for w in r["text"].split()]
        words = [w for w in words if w]
        if not words:
            continue
        kinds = []
        for w in words:
            m = to_money(w)
            if m is not None:
                kinds.append(("money", m))
            elif _INT.match(w):
                kinds.append(("int", int(w)))
            else:
                kinds.append(None)
        others = [w for w, k in zip(words, kinds) if k is None]
        numeric_region = all(len(w) <= 1 for w in others)   # junk like 'Y'
        rest = []
        for w, k in zip(words, kinds):
            if k is not None and (k[0] == "money" or numeric_region):
                nums.append((k[0], k[1], r))
            elif not (numeric_region and len(w) <= 1):
                rest.append(w)
        if rest:
            text_parts.append({**r, "text": " ".join(rest)})
    return text_parts, nums


def find_header(rows):
    """Column-label row: 'Description Qty Price Amount' and variants.

    Accepts a row with a name label plus a numeric label, or one with
    three or more column labels (OCR often garbles 'Item' → 'tem').
    """
    for i, row in enumerate(rows):
        t = _row_text(row)
        if _HDR_NAME.search(t) and _HDR_NUM.search(t):
            return i
        if len(_HDR_ANY.findall(t)) >= 3 and not any(
                to_money(w) is not None for w in t.split()):
            return i
    return None


def _item_block(rows):
    """(start, end) row indexes of the item block."""
    hdr = find_header(rows)
    if hdr is not None:
        start = hdr + 1
    else:
        # no readable header: start after the last metadata row in the
        # upper part of the receipt
        start = 0
        limit = int(len(rows) * 0.7)
        for i, row in enumerate(rows[:limit]):
            t = _row_text(row)
            if _META.search(t) or _DATE.search(t):
                start = i + 1
    end = len(rows)
    for i in range(start, len(rows)):
        if _STOP.search(_row_text(rows[i])):
            end = i
            break
    return start, end


def _clean_name(parts):
    """Join description regions, dropping stand-alone product codes."""
    words = []
    for r in parts:
        t = r["text"].strip()
        if _CODE.match(t.replace(" ", "")) or _INT.match(t):
            continue
        words.append(t)
    return re.sub(r"\s+", " ", " ".join(words)).strip(" :-")


def parse_items(regions, unparsed_out=None):
    """Return a list of dicts:
       {name, code, qty, unit_price, line_total, region, raw_text, confidence}

    If `unparsed_out` is a list, rows inside the item block that contain
    numbers but no readable amount are appended to it.
    """
    rows = group_rows(regions)
    if not rows:
        return []
    start, end = _item_block(rows)

    items = []
    unparsed = []
    pending = None          # last item still allowed to take description rows

    for row in rows[start:end]:
        text_parts, nums = _split_tokens(row)
        moneys = [n for n in nums if n[0] == "money"]
        conf = min(float(r.get("confidence", 1.0)) for r in row)
        row_box = _union([r["bbox_px"] for r in row])

        if moneys:
            line_total = moneys[-1][1]
            unit_price = moneys[-2][1] if len(moneys) >= 2 else None
            qty = None
            # qty = a small integer to the LEFT of the first money token
            first_money_x = moneys[0][2]["bbox_px"][0]
            ints = [n for n in nums if n[0] == "int"
                    and n[2]["bbox_px"][0] <= first_money_x]
            if ints:
                qty = float(ints[-1][1])
            # 'quantity x name' style inline (e.g. "2 x MILK")
            name = _clean_name(text_parts)
            m = re.match(r"^(\d+)\s*[xX*@]\s*(.+)$", name)
            if m and qty is None:
                qty, name = float(m.group(1)), m.group(2).strip()
            # 'name quantity x price' style (e.g. "BREAD 2 x 5.00"): the
            # '2' shares a region with the name, so it is not in `ints`
            # and stayed in the name as 'BREAD 2 x'.
            if m is None and qty is None:
                m = _TRAILING_QTY.match(name)
                if m:
                    qty, name = float(m.group(2)), m.group(1).strip()
            if qty is None:
                qty = 1.0
            if unit_price is None:
                unit_price = round(line_total / qty, 2) if qty else line_total
            # if several prices were printed (S/Price, S/Price) prefer the
            # one consistent with qty × price = amount
            if len(moneys) >= 2:
                for _, v, _ in reversed(moneys[:-1]):
                    if _consistent(qty, v, line_total):
                        unit_price = v
                        break
            # PaddleOCR can merge the Qty and Price columns into one region
            # ('1' + '170.00' → '1170.00' on the OJC demo receipt). Only
            # when no separate qty was read AND the row is inconsistent,
            # try splitting leading digits off the first price; accept the
            # split only if qty × price = amount then holds exactly.
            derived = ""
            if (not ints and m is None and len(moneys) >= 2
                    and not _consistent(qty, unit_price, line_total)):
                split = _split_merged_qty(moneys[0][1], line_total)
                if split:
                    qty, unit_price = split
                    derived = (f"qty/price split from merged OCR token "
                               f"'{moneys[0][1]:.2f}' → {int(qty)} × "
                               f"{unit_price:.2f}")
            code = ""
            codes = [r["text"].strip() for r in text_parts
                     if _CODE.match(r["text"].strip().replace(" ", ""))]
            if codes:
                code = codes[0]
            item = {
                "name": name, "code": code, "qty": qty,
                "unit_price": unit_price, "line_total": line_total,
                "bbox": row_box, "raw_text": _row_text(row),
                "confidence": conf, "derived": derived,
            }
            items.append(item)
            pending = item
        elif nums:
            # numbers but no readable amount: an item row the OCR could
            # not read properly. Do NOT glue it onto the previous item;
            # report it so the UI can send the receipt to review.
            unparsed.append({"raw_text": _row_text(row),
                             "region": box_str(row_box)})
            pending = None
        else:
            # text-only row: description continuation of the item above
            name = _clean_name(text_parts)
            if not name:
                continue
            if pending is not None:
                pending["name"] = (pending["name"] + " " + name).strip() \
                    if pending["name"] else name
                pending["bbox"] = _union([pending["bbox"], row_box])
                pending["raw_text"] += " | " + _row_text(row)
                pending["confidence"] = min(pending["confidence"], conf)
            # a text row ABOVE any numeric row is ignored (column labels etc.)

    if unparsed_out is not None:
        unparsed_out.extend(unparsed)
    out = []
    for it in items:
        name = it["name"] or it["code"]
        if not name:
            continue
        out.append({
            "name": name, "code": it["code"], "qty": it["qty"],
            "unit_price": it["unit_price"], "line_total": it["line_total"],
            "region": box_str(it["bbox"]), "raw_text": it["raw_text"],
            "confidence": it["confidence"], "derived": it["derived"],
        })
    return out


def _consistent(qty, unit_price, line_total):
    if qty is None or unit_price is None:
        return False
    return abs(qty * unit_price - line_total) <= max(0.01, 0.001 * line_total)


def _split_merged_qty(value, line_total):
    """'1170.00' with amount 170.00 → (1.0, 170.0); None if no exact split."""
    whole, frac = f"{value:.2f}".split(".")
    for k in (1, 2):
        if len(whole) <= k:
            break
        q = int(whole[:k])
        rest = float(whole[k:] + "." + frac)
        if q >= 1 and rest > 0 and _consistent(q, rest, line_total):
            return float(q), rest
    return None


def _row_amount(row):
    """Rightmost money token in a row, with its region."""
    _, nums = _split_tokens(row)
    moneys = [n for n in nums if n[0] == "money"]
    if not moneys:
        return None, None
    return moneys[-1][1], moneys[-1][2]


def find_totals(regions):
    """Return {'total', 'total_region', 'subtotal', 'subtotal_region'}.

    total    = amount on the best-ranked TOTAL row whose value is > 0
    subtotal = amount printed on a SUBTOTAL / TOTAL QTY row, if any
    Values are what is PRINTED — never derived from the items.
    """
    rows = group_rows(regions)
    out = {"total": None, "total_region": "", "subtotal": None,
           "subtotal_region": "", "tax": None, "tax_region": "",
           "service_charge": None, "tax_rows": []}

    for row in rows:
        t = _row_text(row)
        if _SUBTOTAL_LABEL.search(t):
            amt, reg = _row_amount(row)
            if amt is not None and amt > 0 and out["subtotal"] is None:
                out["subtotal"] = amt
                out["subtotal_region"] = box_str(_union([r["bbox_px"] for r in row]))

    # The payable total is the BOTTOM-most total-labelled row with an
    # amount > 0. Real PaddleOCR on the OJC demo receipt: 'Total Amount:
    # 170.00' → 'Round Amt' → 'TOTAL: 170.00'; ranking 'TOTAL AMOUNT'
    # first picked the pre-rounding row, so the evidence box pointed at
    # the wrong line (and the value is wrong whenever rounding ≠ 0).
    best = None
    for i_row, row in enumerate(rows):
        t = _row_text(row)
        if not any(p.search(t) for p, _ in _TOTAL_LABELS) \
                or _SUBTOTAL_LABEL.search(t) or _NOT_TOTAL.search(t):
            continue
        amt, reg = _row_amount(row)
        if amt is None or amt <= 0:
            continue
        best = (amt, row, i_row)
    if best:
        out["total"] = best[0]
        out["total_region"] = box_str(_union([r["bbox_px"] for r in best[1]]))
        _find_tax(rows, best[2], out)
    return out


def _find_tax(rows, total_idx, out):
    """Printed tax / service charge ABOVE the payable total row.

    Only rows printed before the total can be part of it; a GST summary
    under the total only restates tax already inside the prices. Rows
    with the same amount are counted once ('SST 6% 3.00' then
    'Total SST 3.00'). Values are what is printed; whether they are added
    on top is decided by the caller (see hybrid_extract._tax_mode).
    """
    amounts, boxes = [], []
    for row in rows[:total_idx]:
        t = _row_text(row)
        if not _TAX_LABEL.search(t) or _TAX_NOT.search(t):
            continue
        amt, _ = _row_amount(row)
        if amt is None or amt <= 0 or amt >= out["total"]:
            continue
        box = box_str(_union([r["bbox_px"] for r in row]))
        kind = "service" if _SERVICE_LABEL.search(t) else "tax"
        out["tax_rows"].append({"amount": amt, "region": box,
                                "raw_text": t, "kind": kind})
        if any(abs(amt - a) < 0.005 for a, _ in amounts):
            continue
        amounts.append((amt, kind))
        boxes.append(box)
    if amounts:
        out["tax"] = round(sum(a for a, _ in amounts), 2)
        service = [a for a, k in amounts if k == "service"]
        out["service_charge"] = round(sum(service), 2) if service else None
        out["tax_region"] = boxes[0]
