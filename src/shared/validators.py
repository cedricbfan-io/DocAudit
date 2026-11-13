"""Schema conformance. Raises early, loudly, with the offending field."""

from config.thresholds_loader import THRESHOLDS


class SchemaError(ValueError):
    pass


def validate_receipt(receipt) -> None:
    # MYR is the project's main scope; receipts in other currencies are
    # read and checked arithmetically, but the MYR price pool is skipped
    # for them (Detector). Unknown codes are still rejected.
    allowed = {THRESHOLDS["currency"]["required"], "CNY", "USD", "SGD",
               "EUR", "GBP"}
    if receipt.currency not in allowed:
        raise SchemaError(f"unsupported currency {receipt.currency!r}")

    if receipt.receipt_total is not None and receipt.receipt_total < 0:
        raise SchemaError(f"negative total: {receipt.receipt_total}")

    for i, item in enumerate(receipt.items):
        if item.qty < 0:
            raise SchemaError(f"item {i} has negative qty")
        if item.unit_price < 0:
            raise SchemaError(f"item {i} has negative unit_price")
        if not item.name or not item.name.strip():
            raise SchemaError(f"item {i} has an empty name")


def validate_anomaly(a) -> None:
    if a.anomaly_type not in ("incorrect_total", "duplicate",
                              "unusual_price"):
        raise SchemaError(f"bad anomaly_type: {a.anomaly_type}")
    if not 0.0 <= a.confidence <= 1.0:
        raise SchemaError(f"confidence out of range: {a.confidence}")
    if not isinstance(a.evidence, dict):
        raise SchemaError("evidence must be a dict")
