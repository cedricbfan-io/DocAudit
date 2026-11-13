"""★ GATE DELIVERABLE: detect.py runs the three checks.

Can start immediately on mock JSON — does not wait for P3's real OCR.
DETERMINISTIC THROUGHOUT. No model in this file, by design.
"""

from src.detect import check_total, check_duplicate, check_price
from src.explain.explain import explain, summary


class Detector:
    def __init__(self, price_pool: dict = None, history: list = None,
                 discount: float = 0.0):
        self.price_pool = price_pool or {}
        self.history = history or []
        self.discount = discount

    def run(self, receipt) -> dict:
        flags = []
        flags += check_total.check(receipt, discount=self.discount)
        dups = check_duplicate.check(receipt, history=self.history)
        # a total skipped by the coverage guard may still be explained by
        # a repeated line (case a): the gap is added to the duplicate
        # warning; otherwise the total stays 'insufficient' (case b)
        flags += check_total.coverage_followup(receipt, dups, self.discount)
        flags += dups
        flags += check_price.check(receipt, pool=self.price_pool)

        receipt.duplicate_flag = any(
            f.anomaly_type == "duplicate" for f in flags)

        explained = explain(flags)
        return {
            "receipt_id": receipt.receipt_id,
            "anomalies": explained,
            "summary": summary(explained),
            "flagged": len(explained) > 0,
            "pool_size": len(self.price_pool),
            # what the receipt-level total check did: 'checked',
            # 'insufficient', 'no_total' or 'mismatch_explained_by_duplicate'
            "total_check": dict(receipt.provenance.get("_total_check") or {}),
        }

    def run_batch(self, receipts: list) -> list:
        return [self.run(r) for r in receipts]
