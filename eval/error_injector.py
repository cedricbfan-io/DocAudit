"""Generates the labelled error set.

No public dataset carries these three anomaly types as ground truth,
so this self-built set is what actually evaluates the system.
Every error receipt is LABELLED.
"""

import copy
import json
import random
from pathlib import Path

random.seed(42)


def inject_wrong_total(receipt, magnitude: float = 2.0):
    r = copy.deepcopy(receipt)
    if r.get("receipt_total") is None:
        return r, {"anomaly_type": "incorrect_total", "expected_delta": 0,
                   "note": "no total to mutate"}
    delta = round(random.choice([-1, 1]) * magnitude, 2)
    r["receipt_total"] = round(r["receipt_total"] + delta, 2)
    return r, {"anomaly_type": "incorrect_total", "expected_delta": delta}


def inject_duplicate_item(receipt):
    r = copy.deepcopy(receipt)
    if not r.get("items"):
        return r, {"anomaly_type": "duplicate", "note": "no items to duplicate"}
    idx = random.randrange(len(r["items"]))
    dup = copy.deepcopy(r["items"][idx])
    r["items"].insert(idx + 1, dup)
    if r.get("receipt_total") is not None:
        r["receipt_total"] = round(r["receipt_total"] + dup["line_total"], 2)
    return r, {"anomaly_type": "duplicate", "duplicated_index": idx}


def inject_inflated_price(receipt, factor: float = 2.5):
    r = copy.deepcopy(receipt)
    if not r.get("items"):
        return r, {"anomaly_type": "unusual_price", "note": "no items to inflate"}
    idx = random.randrange(len(r["items"]))
    item = r["items"][idx]
    old_line = item["line_total"]
    item["unit_price"] = round(item["unit_price"] * factor, 2)
    item["line_total"] = round(item["qty"] * item["unit_price"], 2)
    if r.get("receipt_total") is not None:
        r["receipt_total"] = round(
            r["receipt_total"] - old_line + item["line_total"], 2)
    return r, {"anomaly_type": "unusual_price",
               "inflated_index": idx, "factor": factor}


INJECTORS = {
    "incorrect_total": inject_wrong_total,
    "duplicate": inject_duplicate_item,
    "unusual_price": inject_inflated_price,
}


def build_error_set(normal_paths: list, out_dir: str) -> list:
    """Three faults per normal receipt → ~150 labels from 50 receipts."""
    out = Path(out_dir)
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    labels = []

    for i, path in enumerate(normal_paths):
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
        if not receipt.get("items"):
            continue
        for kind, fn in INJECTORS.items():
            mutated, meta = fn(receipt)
            rid = f"err_{i:03d}_{kind}"
            mutated["receipt_id"] = rid
            (out / "receipts" / f"{rid}.json").write_text(
                json.dumps(mutated, indent=2, ensure_ascii=False),
                encoding="utf-8")
            labels.append({"receipt_id": rid, "anomaly_type": kind,
                           "label": 1, **meta})

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8")
    return labels
