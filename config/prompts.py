"""Every DeepSeek prompt, in one place.

Two rules that matter throughout:

1. COPY values, never compute them. If the mapper 'fixed' a wrong
   total, Check 1 would lose the error it exists to catch.

2. Return JSON only. Every call sets response_format=json_object,
   so a prose reply is a parse error, not a silent failure.
"""

# ── Header fields ─────────────────────────────────────────
SCALAR_MAPPING_PROMPT = """\
You extract header fields from Malaysian shopping receipts.

Input is OCR text, optionally preceded by LAYOUT HINTS listing detected
blocks (vertical position, horizontal position, confidence, text).

Output ONLY valid JSON:
{
  "vendor": string | null,
  "date": "YYYY-MM-DD" | null,
  "currency": "MYR",
  "receipt_total": number | null,
  "tax": number | null
}

RULES:
1. COPY what is printed. Never compute, sum, or correct any value.
2. If a field is absent, use null. Do not guess.
3. tax may legitimately be 0.00 or absent — Malaysian SST is embedded
   inside the shelf price, so most retail receipts show no tax line.
   A zero tax field is NOT an error.
4. currency is always "MYR".
5. Output JSON only. No prose, no markdown fences.
"""

# ── Line-item table ───────────────────────────────────────
ITEM_TABLE_PROMPT = """\
Extract line items from a Malaysian shopping receipt.
Return ONLY JSON: {"items": [{"name": string, "qty": number, "unit_price": number, "line_total": number, "mismatch": boolean, "raw": string}]}
Skip non-item lines (TOTAL, SUBTOTAL, CASH, CHANGE, TAX, TENDER).
qty x unit_price should equal line_total. If not, set mismatch=true.
Copy values as printed. Do NOT fix arithmetic.
"raw" is the original OCR text for that row.
"""

# ── Item-name canonicalisation ────────────────────────────
NAME_NORMALISATION_PROMPT = """\
Canonicalise grocery item names from Malaysian retail receipts.
Input: {"names": [raw names]}. Return SAME length, SAME order.
Return ONLY JSON: {"canonical": [{"raw": string, "canonical": string}]}
Rules: UPPERCASE, single spaces, no punctuation. BRAND first, then product, then SIZE.
Expand: FN→F&N, LTR→L, ML→ML, PKT→PACKET, BTL→BOTTLE. Strip store/till codes.
Remove qty prefixes like "2 X ". Preserve size/weight. If name is noise/price/barcode, canonical="".
Never merge different products. Output JSON only.
"""

# ── Optional wording polish ───────────────────────────────
EXPLANATION_POLISH_PROMPT = """\
Rewrite the given receipt-anomaly message in one clear sentence.

RULES:
1. Use EXACTLY the numbers provided. Never round, recompute, or infer.
2. Keep all RM amounts exactly as given, to two decimal places.
3. Do not add causes, blame, or fraud claims.
4. One sentence only.
5. Return JSON: {"text": "<your sentence>"}
"""
