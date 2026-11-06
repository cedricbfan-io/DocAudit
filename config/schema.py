"""Schema constants — the two frozen JSON contracts.

These mirror the dataclasses in src/shared/models.py and are used
for validation and documentation. The schema is the contract between
extraction and detection: if extraction produces schema-compliant JSON,
detection can run without knowing anything about OCR.
"""

RECEIPT_SCHEMA = {
    "vendor": "string | null",
    "date": "YYYY-MM-DD | null",
    "currency": "MYR",
    "receipt_total": "number | null",
    "tax": "number | null",
    "items": [
        {
            "name": "string",
            "canonical": "string",
            "qty": "number",
            "unit_price": "number",
            "line_total": "number",
            "review": "unreviewed | review_required | corrected | accepted | rejected",
            "reason": "string",
            "region": "string",
            "ocr_confidence": "number",
        }
    ],
    "duplicate_flag": "boolean",
    "_provenance": "object",
    "_needs_review": "boolean",
}

ANOMALY_SCHEMA = {
    "anomaly_type": "incorrect_total | duplicate | unusual_price",
    "reason": "string",
    "evidence": "object",
    "confidence": "number [0, 1]",
}

ALLOWED_ANOMALY_TYPES = ("incorrect_total", "duplicate", "unusual_price")
