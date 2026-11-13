# DocAudit — Project Guide, Issues, and Readiness Checklist

> **Purpose**: This file tells anyone who picks up this project folder exactly what the project is, what works, what is broken, what needs doing, how to run it, and how to judge whether it is production-ready.

---

## 0. Quick start (13/11) — web demo

`python -m src.ui.app` → open **http://127.0.0.1:7860** (opens automatically;
ready when `Running on local URL` is printed). Short guide in Chinese:
`QUICKSTART_zh.md`. The many commands in §2.3 are the batch/CLI pipeline —
they print to the terminal and do not open the UI.

**Review rule (replaces §2.4 "Decision workflow")**: no warning → auto-accepted;
warning → auto-rejected until Accepted; Reject on any row or the TOTAL blocks
the WHOLE receipt from `ledger.csv` and `audit_export.json`. Logic in
`src/ui/review.py`, enforced in `src/ui/export.py` (also for the CLI
`python -m src.ui.cli`).

---

## 1. What This Project Is

DocAudit is an **explainable anomaly detection pipeline for Malaysian (MYR) shopping receipts**. It reads receipt photos with PaddleOCR, structures them with an LLM (DeepSeek via NVIDIA API), runs three deterministic anomaly checks, generates human-readable explanations, and presents everything in a Gradio UI for human review.

**Pipeline**: `Receipt photo → preprocessing → PaddleOCR → LLM header mapping → local column-aware item parser → name normalisation → arithmetic gate → three checks (incorrect total / duplicate / unusual price) → explanation templates → human review → export`

**Dataset**: SROIE (ICDAR 2019 — Scanned Receipt OCR and Information Extraction), accessed via HuggingFace `jsdnrs/ICDAR2019-SROIE`.

---

## 2. How to Run This Project

### 2.1 Prerequisites

- **Python 3.12** (not 3.13 — PaddlePaddle 2.x has no 3.13 wheels; not 3.11 — PaddlePaddle 3.x has a Windows oneDNN bug)
- **Windows** (project developed and tested on Windows 11; Linux/macOS should work but are untested)
- **NVIDIA API key** for DeepSeek (`deepseek-ai/deepseek-v4.1-flash` at `https://integrate.api.nvidia.com/v1`). Get one at https://build.nvidia.com/
- **No GPU required** — PaddleOCR runs on CPU fine

### 2.2 Setup Steps

```bash
# 1. Create a Python 3.12 virtual environment (must use Python 3.12, not 3.13)
C:\Python312\python.exe -m venv C:\venv\da --clear

# 2. Install dependencies using the venv's Python directly
#    (PowerShell activation may not stick — use the venv python path instead)
C:\venv\da\Scripts\python.exe -m pip install setuptools
C:\venv\da\Scripts\python.exe -m pip install -r requirements.txt
C:\venv\da\Scripts\python.exe -m pip install -e .

# 3. Configure API key
copy .env.example .env
# Edit .env and replace the placeholder with your NVIDIA API key:
#   DEEPSEEK_API_KEY=nvapi-your-key-here
#   DOCAUDIT_LLM_EXPLAIN=0

# 4. Verify installation
C:\venv\da\Scripts\python.exe -c "from src.shared import deepseek_client; print(deepseek_client.health_check())"
# Expected: {'ok': True, 'sample': {...}}

# 5. Run tests
C:\venv\da\Scripts\python.exe -m pytest tests/
# Expected: 40 passed
```

### 2.3 Running the Pipeline

```bash
# All commands use the venv Python directly to avoid activation issues

# Extract one receipt (full LLM pipeline — slow, ~300s/image)
C:\venv\da\Scripts\python.exe -m src.extract.extract data/photos/normal/sroie_train_000.jpg

# Extract using hybrid mode (NVIDIA for headers + local parser for items — ~60s/image)
C:\venv\da\Scripts\python.exe -m src.extract.hybrid_extract data/photos/normal/ --gt eval/ground_truth/sroie_gt.json --out data/extracted/normal_extracted.json

# Build the reference pool (needed for Check 3)
C:\venv\da\Scripts\python.exe -m src.detect.reference_builder data/extracted/normal_extracted.json --out data/reference/pool.json

# Generate error-injected evaluation set
C:\venv\da\Scripts\python.exe -m eval.error_injector

# Run detection on all receipts and compute metrics
C:\venv\da\Scripts\python.exe -m eval.compute_metrics

# Launch the Gradio demo UI
C:\venv\da\Scripts\python.exe -m src.ui.app
# Open http://127.0.0.1:7860 in your browser
```

### 2.4 Using the Demo UI

The UI has two tabs:

- **Image (full pipeline)**: Upload a receipt photo → optionally check "Skip API" to bypass the NVIDIA LLM and use rule-based header extraction (much faster, ~16s vs ~60s+) → click "Analyse" → see extracted fields, highlighted evidence regions, anomaly list, and color-coded line-item table.
- **JSON (pre-extracted / error-injected)**: Upload a pre-extracted receipt JSON (from `data/extracted/` or `eval/error_set/receipts/`) → click "Run detection" → see anomaly results without re-running OCR.

**Color-coded table**: The line-item table shows all items plus a TOTAL row at the bottom. Anomaly cells are highlighted in red:
- 🔢 Incorrect total → red on the total value cell (line-level or receipt-level)
- 🔁 Duplicate → red on the item name cell
- 💰 Unusual price → red on the unit price cell

**Decision workflow**: Normal rows are auto-accepted. Only anomaly rows need decisions. Use the dropdown to select an anomaly row, then click Accept (anomaly is real), Reject (false positive), or Correct (fix the value). Click "Accept all anomaly rows" to batch-accept. Click "Export confirmed rows" to save CSV + JSON.

---

## 3. Project Structure

```
DocAudit/
├── .env                          # API keys (NVIDIA DeepSeek)
├── .env.example                  # Template for .env
├── conftest.py                   # pytest path config
├── pyproject.toml                # Package metadata + pytest config
├── requirements.txt              # Python dependencies
├── README.md                     # Quick-start (minimal)
├── PROJECT_REPORT.md             # Full project report
├── PROJECT_GUIDE.md              # THIS FILE
│
├── config/
│   ├── thresholds.yaml           # All tunable thresholds (detection, OCR, LLM)
│   ├── thresholds_loader.py      # Loads YAML into THRESHOLDS dict
│   ├── schema.py                 # Threshold validation schema
│   └── prompts.py                # LLM prompts (simplified for NVIDIA API)
│
├── src/
│   ├── shared/
│   │   ├── models.py             # Receipt, LineItem, Region, Anomaly dataclasses
│   │   ├── arithmetic.py         # line_ok(), receipt_ok(), epsilon(), is_vacuous()
│   │   ├── validators.py         # validate_receipt()
│   │   └── deepseek_client.py    # NVIDIA/DeepSeek API client with cache + retry
│   │
│   ├── extract/
│   │   ├── extract.py            # Full extraction pipeline (extract_one, to_json)
│   │   ├── hybrid_extract.py     # Hybrid: NVIDIA scalars + local item parser
│   │   ├── ocr_engine.py         # PaddleOCR wrapper
│   │   ├── preprocessing.py      # Image cleaning (deskew, denoise, contrast)
│   │   ├── field_mapper.py       # LLM-based field mapping (map_scalars, map_items)
│   │   ├── fallback_extract.py   # Local column-aware parser (no LLM)
│   │   ├── name_normalizer.py    # Rule-based name canonicalisation
│   │   └── cache/                # LLM response cache (auto-created)
│   │
│   ├── detect/
│   │   ├── detect.py             # Detector class — runs all three checks
│   │   ├── check_total.py        # Check 1: incorrect total (arithmetic)
│   │   ├── check_duplicate.py    # Check 2: duplicate (within + across receipts)
│   │   ├── check_price.py        # Check 3: unusual price (modified Z-score)
│   │   ├── similarity.py         # Weighted pair similarity for duplicate check
│   │   └── reference_builder.py  # Builds price reference pool from normal receipts
│   │
│   ├── explain/
│   │   ├── explain.py            # Wraps anomalies with explanation templates
│   │   ├── templates.py          # Human-readable explanation strings
│   │   └── number_guard.py       # Prevents fabricated numbers in explanations
│   │
│   └── ui/
│       ├── app.py                # Gradio demo (two tabs: Image + JSON)
│       ├── export.py             # CSV ledger + JSON audit export
│       └── highlight.py          # Draws evidence boxes on receipt images
│
├── data/
│   ├── photos/
│   │   ├── normal/               # 116 SROIE train images (reference pool source)
│   │   └── test/                 # 4 SROIE test images (held out)
│   ├── extracted/
│   │   ├── normal_extracted.json # 116 extracted normal receipts (51 items)
│   │   └── test_extracted.json   # 4 extracted test receipts
│   ├── reference/
│   │   └── pool.json             # Curated price reference pool (30 products, 140 observations)
│   └── error/                    # (empty — error-injected receipts are in eval/)
│
├── eval/
│   ├── ground_truth/
│   │   └── sroie_gt.json         # SROIE ground truth for all 20 images
│   ├── error_set/
│   │   ├── receipts/             # 48 error-injected JSON files
│   │   └── labels.json           # 48 labels (16 × 3 fault types)
│   ├── results/
│   │   └── metrics.json          # Final evaluation metrics
│   ├── error_injector.py         # Generates error-injected receipts
│   └── compute_metrics.py        # Computes precision/recall/F1/FPR
│
└── tests/
    ├── test_check_total.py       # Check 1 tests (epsilon boundaries)
    ├── test_check_duplicate.py   # Check 2 tests
    ├── test_check_price.py       # Check 3 tests
    ├── test_explain.py           # Explanation template tests
    ├── test_name_normalizer.py   # Name normalisation tests
    └── test_ocr_engine.py        # OCR engine tests
```

---

## 4. Current Evaluation Results

| Check | TP | FP | FN | TN | Precision | Recall | F1 | Status |
|---|---|---|---|---|---|---|---|---|
| **Incorrect total** | 8 | 15 | 19 | 153 | 0.35 | 0.30 | 0.32 | Works, moderate precision (coverage guard added) |
| **Duplicate** | 26 | 0 | 0 | 168 | 1.00 | 1.00 | 1.00 | Perfect |
| **Unusual price** | 11 | 0 | 15 | 168 | 1.00 | 0.42 | 0.59 | Functional (curated reference pool) |

**Review load**: 0.31 (31% of receipts trigger at least one alert)

**Dataset**: 116 SROIE train images (normal), 4 held-out test images, 78 error-injected receipts (26 receipts × 3 fault types)

---

## 5. Known Issues

### 5.1 Critical Issues (Must Fix Before Production)

| ID | Issue | Impact | Fix |
|---|---|---|---|
| **C1** | **Check 3 (unusual price) uses a curated reference pool** | The built-from-receipts pool is empty (SROIE stores have no product overlap). A curated pool of 30 products with realistic Malaysian market prices was created at `data/reference/pool.json`. Check 3 now works (P=1.00, R=0.42) but recall is limited by pool coverage — only 30 of 51 extracted items match pool entries. | For production: replace curated pool with a real market price database, or use a dataset with product overlap. The curated pool is a proof-of-concept, not a production reference. |
| **C2** | **Check 1 (incorrect total) has moderate false positives** | Precision is 0.35 with 15 FP. Root cause: the local parser extracts very few items per receipt (avg 0.44 items/receipt), so `sum(items)` rarely matches `receipt_total`. A coverage guard (skip receipt-level check if `subtotal/total < 0.8 or > 1.2`) was added to mitigate this. | For production: improve item extraction quality (LLM-based or better parser) so the subtotal reliably approximates the receipt total. The coverage guard is a workaround, not a fix. |
| **C3** | **`min_comparables` set to 2** | `thresholds.yaml` has `min_comparables: 2` (lowered from 5). With the curated pool of 30 products (4-5 observations each), this is sufficient. If the pool is replaced with a larger real-world database, raise back to 5. | Keep at 2 for curated pool; raise to 5 when a proper reference database is available. |

### 5.2 Moderate Issues (Should Fix)

| ID | Issue | Impact | Fix |
|---|---|---|---|
| **M1** | **NVIDIA API does not support `response_format={"type":"json_object"}`** | The API returns `None` for this parameter. JSON output must be prompt-driven. `json_mode: false` is set in config, but `deepseek_client.py` still has the conditional code path (lines 89–90). | No code change needed — `json_mode: false` correctly disables it. But if someone switches to the official DeepSeek API (which does support `response_format`), they should set `json_mode: true`. Document this in `.env.example`. |
| **M2** | **Hybrid extraction times out on ~20% of images** | The NVIDIA API call for `map_scalars` occasionally hangs beyond the 180s timeout. These images are backfilled from SROIE ground truth, which is not available for real-world receipts. | Increase timeout to 300s, or add a retry-with-different-prompt strategy. For production use, implement a fallback to the local rule-based parser when the API times out. |
| **M3** | **`map_items()` LLM call is extremely slow and erratic** | The full `extract_one()` pipeline (using `map_items`) takes 50s–25min per image. The hybrid pipeline avoids this, but `extract.py` still calls `field_mapper.map_receipt()` which uses `map_items`. | The hybrid pipeline (`hybrid_extract.py`) is the recommended path. `extract.py`'s `extract_one()` should either be deprecated or updated to use the hybrid approach by default. |
| **M4** | **`check_across()` (cross-receipt duplicate detection) is blocked** | The `mode` in `thresholds.yaml` is `within_receipt`, so cross-receipt duplicate detection never runs. The code exists but is untested against real data. | To enable: set `mode: across_receipts` and provide a `history` list to the Detector. Need to build a history source (e.g., a database of past receipts). |
| **M5** | **No line-item ground truth in SROIE** | SROIE provides header fields (company, date, total) but no per-line-item ground truth. Line-item extraction quality cannot be measured against a gold standard. | Manually annotate ~20 receipts' line items, or use a different dataset that includes line-item labels. |
| **M6** | **Name normalisation is rule-based only** | The LLM-based `canonicalise()` function exists but is not used in the hybrid pipeline (which uses `fallback_canonical()` only). Rule-based normalisation is fast but misses many equivalences (e.g., "COCA COLA 1.5L" vs "COCA-COLA 1.5LTR"). | For better Check 3 performance, enable LLM-based normalisation once the API latency issue is resolved, or improve the rule-based normaliser with more patterns. |

### 5.3 Minor Issues (Nice to Fix)

| ID | Issue | Impact | Fix |
|---|---|---|---|
| **m1** | ~~`data/error/` directory is empty~~ | Removed — error-injected receipts are in `eval/error_set/receipts/`. | ✅ Fixed |
| **m2** | **`src/extract/cache/` directory may contain stale cache files** | LLM response cache can become stale if prompts change. | Run `deepseek_client.clear_cache()` after changing prompts in `config/prompts.py`. |
| **m3** | **README.md is minimal** | Only covers basic setup; does not mention hybrid extraction, the eval pipeline, or the demo UI's JSON tab. | Update README.md or point users to this file. |
| **m4** | ~~`requirements.txt` does not pin exact versions~~ | Pinned: `paddleocr==2.10.0`, `paddlepaddle==2.6.2`. | ✅ Fixed |
| **m5** | **No CI/CD pipeline** | Tests must be run manually. | Add a GitHub Actions workflow that runs `pytest tests/` on Python 3.12. |

---

## 6. What Still Needs to Be Done

### 6.1 Replace Curated Reference Pool with Real Market Data

**Why**: Check 3 currently uses a curated pool of 30 products with hand-crafted Malaysian market prices (`data/reference/pool.json`). This is a proof-of-concept — it proves the Z-score detection works (P=1.00, R=0.42) but the pool is not a legitimate production reference. The built-from-receipts pool is empty because SROIE stores have no product overlap.

**How**:
1. Use a dataset with product overlap (e.g., multiple receipts from the same store chain)
2. Or: integrate a real Malaysian retail price database (e.g., KKM pricing data, GS1 product catalog)
3. Or: use LLM-based item extraction to get more reliable product names, enabling cross-receipt matching

**Expected outcome**: Pool grows from 30 curated products to 100+ real products with observed prices. Recall improves from 0.42 toward 0.70+.

### 6.2 Improve Check 1 Precision (Currently 0.35)

**Why**: The coverage guard (skip receipt-level check if `subtotal/total < 0.8 or > 1.2`) reduced FP from 66→15 but also reduced TP from 25→8. The fundamental issue is that the local parser extracts very few items per receipt (avg 0.44 items/receipt), so the subtotal rarely matches the receipt total.

**Options**:
1. **Improve item extraction** — Use LLM-based extraction (`map_items`) or a better parser to extract all line items, so the subtotal reliably approximates the receipt total. This is the root-cause fix.
2. **Use ground truth totals** — For SROIE, the ground truth provides the correct receipt total. Compare extracted total against ground truth instead of computing from items. (Not applicable for real-world receipts.)
3. **Two-stage check** — Only flag "incorrect total" if line-level checks pass but the total still doesn't match. If line-level checks fail, the total mismatch is likely caused by extraction errors, not a real total error.
4. **OCR confidence weighting** — Suppress total-check alerts when the receipt's average OCR confidence is below 0.80.

### 6.3 Add More Datasets

**Why**: SROIE is the only dataset. Adding more datasets would:
- Provide more receipts for the reference pool
- Test the pipeline on different receipt formats
- Provide line-item ground truth (SROIE lacks this)

**Candidate datasets**:
| Dataset | Source | Line-item GT? | Format |
|---|---|---|---|
| SROIE (current) | HuggingFace `jsdnrs/ICDAR2019-SROIE` | No | Malaysian receipts |
| CORD | HuggingFace `naver-clova-ix/cord-v2` | Yes | Indonesian receipts |
| SROIE test set | HuggingFace `jsdnrs/ICDAR2019-SROIE` split=test | No | Malaysian receipts (held out) |
| Real receipts | Photograph your own | Manual | Any |

**Note**: Adding non-MYR datasets requires either (a) disabling the currency guard (`currency.reject_non_myr: false`) or (b) adding currency conversion. The pipeline is designed for MYR only.

### 6.4 Enable Cross-Receipt Duplicate Detection

**Why**: Currently only within-receipt duplicates are detected (the same item appearing twice on one receipt). Cross-receipt duplicates (the same transaction submitted twice) require a history of past receipts.

**How**:
1. Set `duplicate.mode: across_receipts` in `thresholds.yaml`
2. Build a history source — a JSON file or database of past receipts
3. Pass the history list to `Detector(history=past_receipts)`
4. Tune `threshold_match` and `threshold_review` against labelled data

### 6.5 Improve Name Normalisation

**Why**: The rule-based `fallback_canonical()` normaliser is fast but limited. It misses many product name equivalences, which fragments the reference pool and weakens Check 3.

**How**:
1. Add more pattern rules to `name_normalizer.py` (brand prefixes, size suffixes, unit normalisation)
2. Or: enable the LLM-based `canonicalise()` function in the hybrid pipeline (currently bypassed for speed)
3. Or: use a pre-built product taxonomy (e.g., GS1 GPC) to map raw names to canonical product categories

---

## 7. How to Check If the Product Is Ready

### 7.1 Minimum Viable Product (MVP) Checklist

| # | Criterion | How to Verify | Current Status |
|---|---|---|---|
| 1 | All tests pass | `pytest tests/` → 40 passed | ✅ Pass |
| 2 | Extraction works on real images | `python -m src.extract.hybrid_extract data/photos/normal/` → all 116 extracted | ✅ Pass |
| 3 | Check 1 (incorrect total) detects real errors | `eval/results/metrics.json` → incorrect_total TP > 0 | ✅ Pass (TP=8) |
| 4 | Check 2 (duplicate) detects real errors | `eval/results/metrics.json` → duplicate TP > 0 | ✅ Pass (TP=26, P=R=F1=1.0) |
| 5 | Check 3 (unusual price) detects real errors | `eval/results/metrics.json` → unusual_price TP > 0 | ✅ Pass (TP=11, P=1.0) |
| 6 | Demo UI runs | `python -m src.ui.app` → http://127.0.0.1:7860 loads | ✅ Pass |
| 7 | Explanations are human-readable | UI shows reason text for each anomaly | ✅ Pass |
| 8 | Export works | Click "Export" in UI → CSV + JSON files created | ✅ Pass |
| 9 | LLM health check passes | Click "LLM health check" in UI → "LLM OK" | ✅ Pass |

### 7.2 Production-Ready Checklist

| # | Criterion | How to Verify | Current Status |
|---|---|---|---|
| 1 | Check 3 functional (TP > 0) | Curated pool with 30 products → TP=11, P=1.0, R=0.42 | ✅ Pass (curated pool) |
| 2 | Check 1 precision > 0.80 | Coverage guard added → P=0.35 (was 0.42) | ❌ Not met |
| 3 | Check 1 recall > 0.90 | R=0.30 (was 1.0; coverage guard trades recall for precision) | ❌ Not met |
| 4 | All checks F1 > 0.70 | total=0.32, duplicate=1.0, price=0.59 | ❌ Not met |
| 5 | Review load < 0.30 | Currently 0.31 — close to target | ⚠️ Near |
| 6 | Cross-receipt duplicate detection | Enable `across_receipts` mode with history source | ❌ Not implemented |
| 7 | Handles non-SROIE receipts | Test on real-world receipt photos | ❌ Untested |
| 8 | Handles multi-page receipts | Not implemented — single page only | ❌ Not implemented |
| 9 | API key not committed to git | `.env` is in `.gitignore` | ✅ Pass |
| 10 | Requirements pinned | `requirements.txt` uses `>=` not `==` | ❌ Should pin |
| 11 | CI/CD pipeline | No automated testing | ❌ Not implemented |
| 12 | Documentation complete | README is minimal; this file covers gaps | ⚠️ Partial |

### 7.3 Quick Verification Script

Run this to check the project's current state:

```bash
# 1. Tests
pytest tests/ -q

# 2. LLM connectivity
python -c "from src.shared import deepseek_client; print(deepseek_client.health_check())"

# 3. Data artifacts exist
python -c "
from pathlib import Path
checks = [
    ('Normal images', Path('data/photos/normal').glob('*.jpg')),
    ('Test images', Path('data/photos/test').glob('*.jpg')),
    ('Extracted normal', Path('data/extracted/normal_extracted.json')),
    ('Reference pool', Path('data/reference/pool.json')),
    ('Error set labels', Path('eval/error_set/labels.json')),
    ('Error set receipts', Path('eval/error_set/receipts').glob('*.json')),
    ('Metrics', Path('eval/results/metrics.json')),
    ('Ground truth', Path('eval/ground_truth/sroie_gt.json')),
]
for name, path in checks:
    if hasattr(path, 'exists'):
        ok = path.exists()
    else:
        ok = len(list(path)) > 0
    print(f'  {\"✅\" if ok else \"❌\"} {name}')
"

# 4. Metrics summary
python -c "
import json
m = json.loads(open('eval/results/metrics.json').read())
for k in ['incorrect_total', 'duplicate', 'unusual_price']:
    v = m[k]
    print(f'  {k}: P={v[\"precision\"]}, R={v[\"recall\"]}, F1={v[\"f1\"]}, TP={v[\"tp\"]}, FP={v[\"fp\"]}, FN={v[\"fn\"]}')
print(f'  review_load: {m[\"review_load\"]}')
"
```

---

## 8. Configuration Reference

All thresholds are in `config/thresholds.yaml`. Key settings:

| Setting | Current Value | Purpose | When to Change |
|---|---|---|---|
| `total.tolerance_floor` | 0.01 | Line-level arithmetic epsilon | Lower = stricter line checks |
| `total.receipt_tolerance_floor` | 1.00 | Receipt-level total epsilon | Raise to reduce FP from OCR noise |
| `total.tolerance_relative` | 0.001 | Relative tolerance (0.1% of expected) | Raise for noisier OCR |
| `duplicate.threshold_match` | 0.90 | Score ≥ 0.90 → duplicate | Lower = more aggressive matching |
| `duplicate.threshold_review` | 0.70 | 0.70–0.90 → possible match | Adjust to control review queue size |
| `price.min_comparables` | 5 | Min observations for price check | Lower to 2 for thin data; raise for robustness |
| `price.modified_z_threshold` | 3.5 | Flag if |Z| > 3.5 | Lower = more sensitive to price anomalies |
| `deepseek.timeout` | 180 | API call timeout (seconds) | Raise if API is slow; lower to fail faster |
| `deepseek.json_mode` | false | Use `response_format` JSON mode | Set true for official DeepSeek API (not NVIDIA) |
| `deepseek.cache` | true | Cache LLM responses to disk | Keep true for reproducible metrics |
| `ocr.confidence.review_below` | 0.80 | Flag items with OCR conf < 0.80 | Lower to reduce review load |

---

## 9. Key Design Decisions and Their Rationale

1. **Hybrid extraction (NVIDIA for headers, local parser for items)**: The NVIDIA API is stable for `map_scalars` (~40s) but erratic for `map_items` (50s–25min). The local column-aware parser is fast and deterministic for line items. Combining them cuts per-image time from ~300s to ~60s.

2. **Line-level vs receipt-level tolerance split**: Line-level arithmetic (qty × price = total) should be tight (0.01) because OCR reads each field independently. Receipt-level total comparison should be loose (1.00) because OCR misses on individual lines accumulate.

3. **Three-state duplicate detection (match / possible / non-match)**: Binary accept/reject loses the distinction between "system found a real error" and "system invented one". The middle state routes to human review, which is what generates the correction vs rejection data needed to tune the system.

4. **Error injection on JSON, not images**: Mutating extracted receipt JSON (wrong total, duplicate item, inflated price) is fast and deterministic. Photo editing would be slow, non-reproducible, and would test OCR rather than detection.

5. **Deterministic detection (no ML in checks)**: All three checks use arithmetic, statistics, and string similarity — no models. This makes results reproducible and explainable. The LLM is used only for extraction (text → structure), never for detection.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Reference pool** | A dictionary mapping canonical product names → list of observed unit prices. Built from "normal" receipts. Used by Check 3. |
| **Modified Z-score** | A robust outlier statistic: `Mᵢ = 0.6745 × (xᵢ − median) / MAD`. More resistant to outliers than standard Z-score. |
| **MAD** | Median Absolute Deviation: `median(|xᵢ − median|)`. A robust measure of spread. |
| **Arithmetic gate** | Step ⑤ of extraction. Checks `qty × unit_price = line_total` for each item, and `subtotal + tax − discount = receipt_total` for the receipt. Failures become `review_required`. |
| **Provenance** | Metadata attached to each receipt recording where each field came from (OCR, LLM, ground truth, rules). Enables debugging and audit trails. |
| **Review load** | Fraction of receipts that trigger at least one alert. Lower = better (less human work). Target: < 0.30. |
| **SROIE** | ICDAR 2019 Scanned Receipt OCR and Information Extraction competition dataset. 626 train + 347 test images of Malaysian receipts. |
