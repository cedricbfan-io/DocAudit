"""Full evaluation pipeline: build pool → inject errors → detect → score.

Run after extraction completes:
    python codes/run_full_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.models import Receipt, LineItem
from src.detect.reference_builder import build_pool, save_pool, pool_stats
from src.detect.detect import Detector
from eval.error_injector import build_error_set
from eval.compute_metrics import metrics, write_report


def receipt_from_json(data: dict) -> Receipt:
    items = []
    for i in data.get("items", []):
        items.append(LineItem(
            name=i["name"],
            qty=i["qty"],
            unit_price=i["unit_price"],
            line_total=i["line_total"],
            canonical=i.get("canonical", ""),
            region=i.get("region", ""),
            ocr_confidence=i.get("ocr_confidence", 1.0),
            review=i.get("review", "unreviewed"),
            reason=i.get("reason", ""),
        ))
    return Receipt(
        vendor=data.get("vendor"),
        date=data.get("date"),
        currency=data.get("currency", "MYR"),
        items=items,
        receipt_total=data.get("receipt_total"),
        tax=data.get("tax"),
        receipt_id=data.get("receipt_id", ""),
        image_path=data.get("image_path", ""),
        provenance=data.get("_provenance", {}),
    )


def run_full_eval():
    print("=" * 60)
    print("DocAudit Full Evaluation Pipeline")
    print("=" * 60)

    # 1. Load extracted receipts
    print("\n[1/6] Loading extracted receipts...")
    extracted_path = Path("data/extracted/normal_extracted.json")
    if not extracted_path.exists():
        print(f"  ERROR: {extracted_path} not found. Run extraction first.")
        return
    raw_receipts = json.loads(extracted_path.read_text(encoding="utf-8"))
    valid = [r for r in raw_receipts if "error" not in r]
    errors = [r for r in raw_receipts if "error" in r]
    print(f"  Loaded {len(valid)} valid receipts, {len(errors)} errors")
    if errors:
        for e in errors[:5]:
            print(f"    ERROR: {e.get('image', '?')}: {e.get('error', '?')[:80]}")

    receipts = [receipt_from_json(r) for r in valid]
    total_items = sum(len(r.items) for r in receipts)
    print(f"  Total line items across all receipts: {total_items}")

    # 2. Build reference pool
    print("\n[2/6] Building reference pool...")
    pool = build_pool(receipts)
    stats = pool_stats(pool)
    print(f"  Built pool: {stats['items']} products, {stats['observations']} observations")

    if not pool:
        curated_path = Path("data/reference/pool.json")
        if curated_path.exists():
            pool = json.loads(curated_path.read_text(encoding="utf-8"))
            print(f"  Built pool empty — loaded curated pool from {curated_path}")
            print(f"  Curated pool: {len(pool)} products, "
                  f"{sum(len(v) for v in pool.values())} observations")
        else:
            print(f"  WARNING: {stats.get('warning', 'empty pool')}")

    save_pool(pool)
    print(f"  Saved to data/reference/pool.json")

    # Show top products by observation count
    sorted_pool = sorted(pool.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"  Top 10 products by observation count:")
    for name, prices in sorted_pool[:10]:
        print(f"    {name[:50]}: {len(prices)} obs, prices={prices[:5]}")

    # 3. Generate error-injected set
    print("\n[3/6] Generating error-injected set...")
    # Split receipts into individual JSON files for the injector
    tmp_dir = Path("eval/tmp_normal")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    normal_paths = []
    for i, r in enumerate(valid):
        p = tmp_dir / f"normal_{i:03d}.json"
        p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        normal_paths.append(str(p))

    labels = build_error_set(normal_paths, "eval/error_set")
    print(f"  Generated {len(labels)} error-injected receipts (3 fault types × {len(valid)} normal)")

    # 4. Run detection on all receipts
    print("\n[4/6] Running detection on error + clean receipts...")
    detector = Detector(price_pool=pool)

    preds = []
    # Error-set receipts
    error_receipts_dir = Path("eval/error_set/receipts")
    error_files = sorted(error_receipts_dir.glob("*.json"))
    for f in error_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        receipt = receipt_from_json(data)
        result = detector.run(receipt)
        anomaly_types = set()
        for a in result["anomalies"]:
            atype = a["anomaly_type"]
            anomaly_types.add(atype)
            preds.append({"receipt_id": data["receipt_id"],
                          "anomaly_type": atype, "label": 1})
        # Also record non-detections for labelled anomaly types
        for atype in ("incorrect_total", "duplicate", "unusual_price"):
            if atype not in anomaly_types:
                preds.append({"receipt_id": data["receipt_id"],
                              "anomaly_type": atype, "label": 0})

    # Clean receipts (negatives)
    for r in valid:
        receipt = receipt_from_json(r)
        result = detector.run(receipt)
        anomaly_types = set()
        for a in result["anomalies"]:
            atype = a["anomaly_type"]
            anomaly_types.add(atype)
            preds.append({"receipt_id": r["receipt_id"],
                          "anomaly_type": atype, "label": 1})
        for atype in ("incorrect_total", "duplicate", "unusual_price"):
            if atype not in anomaly_types:
                preds.append({"receipt_id": r["receipt_id"],
                              "anomaly_type": atype, "label": 0})

    total_receipts = len(error_files) + len(valid)
    print(f"  Ran detection on {len(error_files)} error + {len(valid)} clean = {total_receipts} receipts")
    print(f"  Total predictions: {len(preds)}")

    # 5. Compute metrics
    print("\n[5/6] Computing metrics...")
    m = metrics(preds, labels, total_receipts)
    write_report(m)
    print(f"  Saved to eval/results/metrics.json")

    # 6. Print results
    print("\n[6/6] Results:")
    print("-" * 60)
    print(f"{'Check':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 60)
    for atype in ("incorrect_total", "duplicate", "unusual_price"):
        v = m[atype]
        print(f"{atype:<20} {v['tp']:>4} {v['fp']:>4} {v['fn']:>4} {v['tn']:>4} "
              f"{v['precision']:>6.2f} {v['recall']:>6.2f} {v['f1']:>6.2f}")
    print("-" * 60)
    print(f"Review load: {m['review_load']:.4f}")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    run_full_eval()
