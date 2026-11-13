"""Download more SROIE train images and generate ground truth entries.

Scales the normal set from 16 to 116 receipts to populate the reference pool
with enough repeated products for Check 3 (unusual price) to function.
"""

import json
from pathlib import Path
from datasets import load_dataset


def download_and_gen_gt(num_existing=16, num_new=100):
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="train")

    photo_dir = Path("data/photos/normal")
    photo_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path("eval/ground_truth/sroie_gt.json")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    existing_ids = {e["receipt_id"] for e in gt["normal"]}

    new_entries = []
    for i in range(num_existing, num_existing + num_new):
        rid = f"sroie_train_{i:03d}"
        if rid in existing_ids:
            continue

        entry = ds[i]
        img = entry["image"]
        img.save(str(photo_dir / f"{rid}.jpg"))

        gt_entry = {
            "receipt_id": rid,
            "company": entry.get("company", ""),
            "date": entry.get("date", ""),
            "total": entry.get("total", ""),
            "address": entry.get("address", ""),
        }
        new_entries.append(gt_entry)
        print(f"  Downloaded {rid}: {gt_entry['company'][:40]}, total={gt_entry['total']}")

    gt["normal"].extend(new_entries)
    gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDownloaded {len(new_entries)} new images.")
    print(f"Normal set now: {len(gt['normal'])} receipts")
    print(f"Ground truth updated: {gt_path}")


if __name__ == "__main__":
    download_and_gen_gt()
