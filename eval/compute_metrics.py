"""The four metrics.

    Precision = TP/(TP+FP)      Recall = TP/(TP+FN)
    FPR       = FP/(FP+TN)      Review load = Alerts/Total receipts

Run AFTER extraction is cached — that is what makes the numbers stable.
"""

import json
import pathlib


def confusion(preds: list, labels: list, atype: str) -> dict:
    """Counts for ONE anomaly type.

    Labelled positives the system never flagged are counted as false
    negatives — otherwise recall is inflated by omission.
    """
    truth = {(l["receipt_id"], l["anomaly_type"]): l["label"] for l in labels}
    tp = fp = fn = tn = 0
    seen = set()

    for p in preds:
        if p["anomaly_type"] != atype:
            continue
        key = (p["receipt_id"], atype)
        seen.add(key)
        actual = truth.get(key, 0)
        if p["label"] == 1 and actual == 1:
            tp += 1
        elif p["label"] == 1 and actual == 0:
            fp += 1
        elif p["label"] == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    for key, actual in truth.items():
        if actual == 1 and key[1] == atype and key not in seen:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(preds: list, labels: list, total_receipts: int) -> dict:
    out = {}
    for atype in ("incorrect_total", "duplicate", "unusual_price"):
        c = confusion(preds, labels, atype)
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[atype] = {"precision": round(prec, 4), "recall": round(rec, 4),
                      "fpr": round(fpr, 4), "f1": round(f1, 4), **c}

    alerts = sum(1 for p in preds if p["label"] == 1)
    out["review_load"] = round(alerts / max(total_receipts, 1), 4)
    return out


def write_report(m: dict, path: str = "eval/results/metrics.json"):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return str(p)
