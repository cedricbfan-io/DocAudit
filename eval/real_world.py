"""Does DocAudit find the TOTAL on receipts it was not tuned on?

    python -m eval.real_world sroie      # SROIE photos vs the dataset's own total
    python -m eval.real_world varied     # 20 rendered layouts with known totals
    python -m eval.real_world all

Both run the APP's pipeline (extract_hybrid: preprocessing → PaddleOCR,
English first, Chinese model as fallback → receipt_parser) — only the OCR
call is cached in eval/.ocr_cache/ so re-runs after a parser change take
seconds. Results go to eval/results/total_accuracy.json (a new file; the
detection metrics in eval/results/metrics.json are untouched).

What is measured (team-measured, not from a paper):
  total exact match = the receipt_total the app reads equals the expected
  total to the cent; 'not found' = no total; 'wrong' = a different number.
  For 'varied', a receipt that prints no total counts as correct only if
  the app also reports none.
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract.hybrid_extract import extract_hybrid  # noqa: E402
from src.extract.ocr_engine import run_ocr  # noqa: E402
from src.detect.detect import Detector  # noqa: E402

CACHE = ROOT / "eval" / ".ocr_cache"
VARIED = ROOT / "eval" / "varied"
RESULTS = ROOT / "eval" / "results" / "total_accuracy.json"


def cached_ocr(path, lang=None):
    """run_ocr with a cache keyed by image bytes + language."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]
    f = CACHE / f"{key}_{lang or 'default'}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    regions = run_ocr(path, lang=lang)
    f.write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")
    return regions


def _run(image, tmp):
    """The app's extraction on a copy (preprocessing writes next to it)."""
    copy = Path(tmp) / image.name
    shutil.copy(image, copy)
    return extract_hybrid(str(copy), skip_api=True, ocr=cached_ocr)


def _score(rows):
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    from collections import Counter
    return {"n": n, "exact": ok, "accuracy": round(ok / n, 4) if n else None,
            "receipts_with_warning": sum(bool(r.get("warnings")) for r in rows),
            "total_check": dict(Counter(r.get("total_check") for r in rows)),
            "not_found": sum(r["found"] is None and r["expected"] is not None
                             for r in rows),
            "wrong": sum(not r["ok"] and r["found"] is not None for r in rows)}


def eval_sroie():
    gt = json.loads((ROOT / "eval/ground_truth/sroie_gt.json").read_text())
    truth = {g["receipt_id"]: g.get("total") for s in gt.values() for g in s}
    from src.ui.app import _load_pool
    pool = _load_pool()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for img in sorted((ROOT / "data/photos").glob("*/*.jpg")):
            if img.stem.endswith("_clean") or img.stem not in truth:
                continue
            try:
                exp = round(float(str(truth[img.stem]).replace(",", "")), 2)
            except (TypeError, ValueError):
                continue
            r = _run(img, tmp)
            res = Detector(price_pool=pool).run(r)
            got = r.receipt_total
            rows.append({"receipt": img.stem, "expected": exp, "found": got,
                         "ok": got is not None and abs(got - exp) < 0.005,
                         "ocr_lang": r.provenance.get("_ocr_lang"),
                         "total_check": res["total_check"].get("status"),
                         # SROIE receipts are genuine: any warning is a false alarm
                         "warnings": sorted({a["anomaly_type"]
                                             for a in res["anomalies"]})})
    return rows


def eval_varied():
    expected = json.loads((VARIED / "expected.json").read_text(encoding="utf-8"))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, exp in sorted(expected.items()):
            r = _run(VARIED / name, tmp)
            res = Detector().run(r)
            got = r.receipt_total
            ok = (got is None) if exp is None else (
                got is not None and abs(got - exp) < 0.005)
            rows.append({"receipt": name, "expected": exp, "found": got, "ok": ok,
                         "total_check": res["total_check"].get("status"),
                         "items": len(r.items), "currency": r.currency,
                         "ocr_lang": r.provenance.get("_ocr_lang"),
                         "warnings": [a["anomaly_type"] for a in res["anomalies"]]})
    return rows


def main(argv=None):
    what = (argv or sys.argv[1:] or ["all"])[0]
    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for name, fn in (("sroie", eval_sroie), ("varied", eval_varied)):
        if what not in (name, "all"):
            continue
        rows = fn()
        s = _score(rows)
        out[name] = {"summary": s, "rows": rows}
        print(f"{name}: total exact match {s['exact']}/{s['n']} "
              f"({s['accuracy']:.1%}) · not found {s['not_found']} · wrong {s['wrong']}"
              f" · receipts with a warning {s['receipts_with_warning']}"
              f" · total check {s['total_check']}")
        for r in rows:
            if not r["ok"]:
                print(f"   ✗ {r['receipt']}: expected {r['expected']} found {r['found']}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
