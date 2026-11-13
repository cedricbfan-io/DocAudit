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
_CUR_PREFIX = r"(?:RM|RH|MYR|USD|US\$|S\$|HK\$|RMB|CNY|\$|¥|￥|€|£)"
_CUR_SUFFIX = r"(?:RM|MYR|USD|RMB|CNY|元|\$)"
_MONEY = re.compile(
    rf"^(?:{_CUR_PREFIX}\s*-?\d{{1,3}}(?:,\d{{3}})+\.\d{{2}}"      # RM1,234.50
    rf"|-?\d{{1,3}}(?:,\d{{3}})+\.\d{{2}}"                          # 1,234.50
    rf"|{_CUR_PREFIX}?\s*-?\d{{1,6}}[.,:]\d{{2}}{_CUR_SUFFIX}?"        # 17.00 / 17,00
    rf"|-{_CUR_PREFIX}\d{{1,6}}[.,:]\d{{2}}"                            # -RM0.02
    rf"|{_CUR_PREFIX}\s*-?\d{{1,6}}[.,]\d)$",                         # ¥17.4
    re.IGNORECASE)
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
                r"TOTAL\s*SALES|AMOUNT\s*DUE|BALANCE\s*DUE|TOTAL\s*DUE|"
                r"AMOUNT\s*PAYABLE|TOTAL\s*PAYABLE|TOTAL\s*\(?RM\)?|AMT\s*DUE|"
                r"JUMLAH|TTL)\b", re.I), "total"),
    # OCR often reads TOTAL as 'Tota1' / 'TotaI' (SROIE train_088)
    (re.compile(r"\bTOTA[L1I]?\b|\bTOTA[L1I]\w*", re.I), "total"),
    # Chinese receipts (needs the Chinese OCR model): 合计 总计 应付 实付…
    (re.compile(r"(合\s*计|总\s*计|总\s*额|应\s*付|实\s*付|应\s*收|"
                r"付款金额|支付金额|消费金额|订单金额|合計|總計)"), "total"),
]
# rows that are payment / change lines: the payable TOTAL is printed
# above them, and a 'Total:' row below them belongs to a tax summary
_PAYMENT = re.compile(r"\b(CASH|CHANGE|TENDER\w*|VISA|MASTER\w*|CARD|DEBIT|"
                      r"CREDIT|PAID|PAYMENT|BAKI|TUNAI|AMEX|E-?WALLET)\b|"
                      r"找\s*零|现\s*金|实\s*收|收\s*款", re.I)
# a tax summary table printed under the total ('GST Summary', 'GST Analysis')
_SUMMARY = re.compile(r"(GST|SST|TAX)\s*(SUM+[AE]?RY|ANALYSIS)", re.I)
_ROUNDING = re.compile(r"ROUND\w*|ADJ\w*", re.I)
# 'Inclusive' / OCR 'lnclusive' / 'Inci', and 'Total with GST'
_INCLUSIVE = re.compile(r"\b[IL1]NC\w*|\bWITH\s+(GST|SST|TAX)", re.I)
_LETTERS = re.compile(r"[A-Za-z\u4e00-\u9fff]")
_TRAILING_QTY = re.compile(r"^(.*\S)\s+(\d{1,3})\s*[xX*@]$")
_TRAILING_XQTY = re.compile(r"^(.*\S)\s*[xX×*]\s*(\d{1,3})$")
# '2 APPLE' — or glued by OCR, '1PEAR' / '2Bur-BRC': one digit followed by
# a word of 3+ letters (so '7UP', '3M', '100PLUS' stay names)
_LEADING_QTY = re.compile(r"^(\d{1,2})\s+([A-Za-z\u4e00-\u9fff].*)$|"
                          r"^(\d)([A-Za-z]{3,}.*)$")
# names that start with a number but are not a quantity ('3 IN 1 COFFEE')
_NOT_QTY_NAME = re.compile(r"^\d+\s*(IN\s*1|PLY|PCS|PC|KG|G|ML|L|CM|MM|PACK|"
                           r"PKT|SHEETS?|PAIRS?)\b", re.I)

# printed tax / service-charge rows. Rows that only MENTION tax (a GST
# summary table, 'Total incl. GST', 'GST Reg No', 'Tax Invoice') are not
# an amount added on top.
_TAX_LABEL = re.compile(r"\b(SST|GST|TAX|SERVICE\s*CH\w*|SVC\s*CHG|CUKAI)\b", re.I)
_SERVICE_LABEL = re.compile(r"\b(SERVICE\s*CH\w*|SVC\s*CHG)\b", re.I)
_TAX_NOT = re.compile(r"\b(INCL\w*|EXCL\w*|SUMMARY|REG\w*|INVOICE|ID|NO|"
                      r"CODE|ROUND\w*)\b", re.I)

_SUBTOTAL_LABEL = re.compile(r"\b(SUB\s*-?\s*TOTAL|SUBTTL)\b|TOTA[L1I]\s*QTY|"
                             r"小\s*计|数量合计", re.I)
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
    neg = t.startswith("-")
    t = re.sub(rf"^-?{_CUR_PREFIX}", "", t, flags=re.I)
    if neg and not t.startswith("-"):
        t = "-" + t
    t = re.sub(rf"{_CUR_SUFFIX}$", "", t, flags=re.I)
    if re.search(r",\d{3}(?:[.]|$)", t):   # thousands separators
        t = t.replace(",", "")
    m = re.match(r"^(-?\d+)[.,:](\d{1,2})$", t)
    if not m:
        return None
    t = f"{m.group(1)}.{m.group(2)}"    # ',' or '.' as the decimal mark
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
        # '@1.89' (unit price after '@') is a price, not a name word
        words = [w.strip(":=_;|.\u2014\"'°@") for w in r["text"].split()]
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

    block = rows[start:end]
    carry = None            # name printed on the line ABOVE its qty/amount line
    for k, row in enumerate(block):
        text_parts, nums = _split_tokens(row)
        moneys = [n for n in nums if n[0] == "money"]
        conf = min(float(r.get("confidence", 1.0)) for r in row)
        row_box = _union([r["bbox_px"] for r in row])

        # a trailing 0.00 after a real amount is a discount / tax column,
        # not the line amount (SROIE train_065: '... 35.00 0.00')
        while len(moneys) >= 2 and moneys[-1][1] == 0 and moneys[-2][1] > 0:
            moneys = moneys[:-1]
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
            if not name and carry:
                # 'CHOC BAR' / '2 x 2.50   5.00': the name is the line above
                name, carry_row = carry
                row_box = _union([row_box, _union([r["bbox_px"] for r in carry_row])])
            carry = None
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
            # 'name x1' (Chinese delivery slips: '鸡肉云吞 x1 ¥17.4')
            if m is None and qty is None:
                m = _TRAILING_XQTY.match(name)
                if m:
                    qty, name = float(m.group(2)), m.group(1).strip()
            # 'qty name amount' ('2 APPLE 1.00', '2 Bur-BRC @ 1.89 3.78'):
            # the leading number shares the name's OCR region
            if m is None and qty is None:
                m = _LEADING_QTY.match(name)
                if m and not _NOT_QTY_NAME.match(name):
                    q, rest = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
                    qty, name = float(q), rest.strip(" @")
                else:
                    m = None
            # 'BREAD 2 @ 5.00 10.00': '@' is dropped as punctuation, so a
            # trailing number counts as the quantity only if it makes
            # qty × price = amount hold exactly
            if m is None and qty is None and len(moneys) >= 2:
                m2 = re.match(r"^(.*\S)\s+(\d{1,3})$", name)
                if m2 and _consistent(float(m2.group(2)), moneys[-2][1], line_total):
                    qty, name = float(m2.group(2)), m2.group(1)
                    m = m2
            name = name.rstrip(" @").strip()
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
            # A product code read as the quantity ('70549 1 6.00 6.00' →
            # qty 801 on SROIE train_034): a big 'quantity' that breaks the
            # line while qty 1 fits it exactly is a code, not a quantity.
            derived_code = ""
            if (ints and qty and qty >= 10 and len(moneys) >= 2
                    and not _consistent(qty, unit_price, line_total)
                    and _consistent(1.0, moneys[-2][1], line_total)):
                derived_code = (f"'{int(qty)}' read as quantity breaks "
                                f"{int(qty)} × {moneys[-2][1]:.2f} = {line_total:.2f}; "
                                f"treated as a code, qty 1")
                qty, unit_price = 1.0, moneys[-2][1]
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
            derived = derived or derived_code
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
            # An unlabelled subtotal ('Food 12.06' under two items that add
            # up to 12.06) ends the item block; it is not another item.
            if (len(items) >= 2 and not ints and m is None and len(moneys) == 1
                    and abs(sum(i["line_total"] for i in items) - line_total) < 0.005):
                break
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
            # text-only row: description continuation of the item above —
            # unless the NEXT row is an amount line with no name of its own,
            # in which case this text is that item's name
            name = _clean_name(text_parts)
            if not name:
                continue
            if k + 1 < len(block):
                # only a bare amount line ('2 x 2.50  5.00'): rows with a
                # product code or any text keep SROIE's name-below layout
                n_text, n_nums = _split_tokens(block[k + 1])
                if any(x[0] == "money" for x in n_nums) and not n_text:
                    carry = (name, row)
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

    # The payable TOTAL is the bottom-most total-labelled row with an
    # amount > 0 — but (a) not inside a GST/tax summary table and (b) not
    # below the first payment line (CASH / CHANGE / VISA…) when a total
    # was already printed above it. SROIE before 13/11: picking the
    # bottom-most row took the summary's 'Total: 106.10 6.37' (the tax
    # column) instead of 'Total (RM) 112.45' in 28 of 107 receipts.
    end = len(rows)
    first_total = next((i for i, row in enumerate(rows)
                        if _is_total_label(_row_text(row))), len(rows))
    for i in range(first_total, len(rows)):
        if _SUMMARY.search(_row_text(rows[i])):
            end = i
            break
    cands = []
    for i_row in range(end):
        row = rows[i_row]
        t = _row_text(row)
        if not _is_total_label(t):
            continue
        amt, row_used = _row_amount(row), row
        if amt[0] is None and i_row + 1 < end:
            # label and amount on separate lines ('TOTAL' / 'RM 27.35')
            nxt = rows[i_row + 1]
            n_amt = _row_amount(nxt)
            if n_amt[0] is not None and not _LETTERS.search(
                    re.sub(_CUR_PREFIX, "", _row_text(nxt), flags=re.I)):
                amt, row_used = n_amt, row + nxt
        if amt[0] is None or amt[0] <= 0:
            continue
        cands.append((i_row, amt[0], row_used))
    if cands:
        first_pay = next((i for i in range(cands[0][0] + 1, end)
                          if _PAYMENT.search(_row_text(rows[i]))
                          and not _is_total_label(_row_text(rows[i]))), None)
        if first_pay is not None:
            cands = [c for c in cands if c[0] < first_pay] or cands
        i_row, amt, row_used = cands[-1]
        amt, row_used, i_row = _after_rounding(rows, i_row, amt, row_used, end)
        out["total"] = amt
        out["total_region"] = box_str(_union([r["bbox_px"] for r in row_used]))
        _find_tax(rows, i_row, out)
    return out


def _is_total_label(t):
    """A payable-total row label (not subtotal, tax, discount, payment).

    'Total Sales (Inclusive of GST)' IS a total; 'Total GST' is not.
    """
    if not any(p.search(t) for p, _ in _TOTAL_LABELS):
        return False
    if _SUBTOTAL_LABEL.search(t) or re.search(r"\bEXCL\w*", t, re.I):
        return False
    if re.search(r"TOTA[L1I]?\b.*ROUNDED|TOTA[L1I]\s*ROUND\w*|ROUNDED\s*TOTAL",
                 t, re.I) and \
            not re.search(r"\b(CASH|CHANGE)\b", t, re.I):
        return True              # 'TOTAL ROUNDED 33.90' is the payable total
    if _NOT_TOTAL.search(t) and not (
            _INCLUSIVE.search(t) and not re.search(
                r"\b(CASH|CHANGE|DISCOUNT|DISKAUN|ROUND\w*)\b", t, re.I)):
        return False
    if re.search(r"TOTAL\s*(GST|SST|TAX)\b", t, re.I) and not _INCLUSIVE.search(t):
        return False
    return True


def _after_rounding(rows, i_row, amt, row_used, end):
    """'TOTAL 60.31' → 'ROUNDING -0.01' → 'RM 60.30': the payable amount
    is the rounded one, even when its row has no TOTAL label (SROIE
    train_001). Accept it only if total + rounding = that amount."""
    for j in range(i_row + 1, min(i_row + 3, end)):
        t = _row_text(rows[j])
        if not _ROUNDING.search(t) or _PAYMENT.search(t):
            continue
        r_amt, _ = _row_amount(rows[j])
        if r_amt is None:
            continue
        for k in range(j + 1, min(j + 3, end)):
            tk = _row_text(rows[k])
            if _PAYMENT.search(tk) or _ROUNDING.search(tk):
                continue
            k_amt, _ = _row_amount(rows[k])
            if k_amt is not None and abs(round(amt + r_amt, 2) - k_amt) < 0.005:
                return k_amt, rows[k], k
        break
    return amt, row_used, i_row


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


def infer_total(regions, items_sum: float, below_y: float = 0):
    """No TOTAL label anywhere (US slip: 'Food 12.06 / Tax 1.06 /
    Payment 13.12 / Change Due 0.13'): look for a printed amount below the
    items that equals items + printed tax (or items alone).

    Returns {'total', 'total_region', 'raw_text', 'tax'} or None. The
    value is still what is PRINTED on that row — but because it was
    chosen BY the arithmetic, the total check cannot confirm it; callers
    must mark it for review (check_total status 'inferred').
    """
    if not items_sum:
        return None
    rows = group_rows(regions)
    tax = 0.0
    for row in rows:
        tt = _row_text(row)
        if _TAX_LABEL.search(tt) and not _TAX_NOT.search(tt):
            a, _ = _row_amount(row)
            if a and 0 < a < items_sum:
                tax += a
    # with printed tax the total must be items + tax (a row equal to the
    # items alone is then the subtotal, e.g. 'Food 12.06')
    target = round(items_sum + tax, 2)
    for row in rows:
        if min(r["bbox_px"][1] for r in row) <= below_y:
            continue                 # an item row is not the total
        tt = _row_text(row)
        if re.search(r"\bCHANGE|找\s*零|\bBAKI\b|\bTAX\b|GST|SST", tt, re.I) \
                or _SUBTOTAL_LABEL.search(tt):
            continue
        a, _ = _row_amount(row)
        if a is not None and abs(a - target) < 0.005:
            return {"total": a, "raw_text": tt, "tax": round(tax, 2),
                    "total_region": box_str(_union([r["bbox_px"] for r in row]))}
    return None
