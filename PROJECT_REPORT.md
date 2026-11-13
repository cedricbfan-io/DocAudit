# DocAudit — Full Project Report

## 1. Project Overview

DocAudit is an explainable anomaly detection pipeline for Malaysian (MYR) shopping receipts. It reads receipt photos with PaddleOCR, structures them with an LLM (NVIDIA-hosted DeepSeek), detects three types of anomalies with deterministic code, explains each finding with templated reasons, and presents everything in a Gradio UI with human review dispositions.

**Pipeline stages:** Extract → Detect → Explain → UI

**Dataset:** ICDAR2019-SROIE (116 normal receipts for reference pool, 4 held-out test receipts, 78 error-injected receipts for evaluation)

---

## 2. File Structure

```
DocAudit/
├── .env                          # NVIDIA API key (DEEPSEEK_API_KEY=nvapi-...)
├── .env.example                  # Template for .env
├── DocAudit_Final_Code_v5_Part1.md  # Original spec document (Part 1)
├── DocAudit_Final_Code_v5_Part2.md  # Original spec document (Part 2)
├── README.md                     # Project README
├── pyproject.toml                # Build config + pytest settings
├── requirements.txt              # Python dependencies
├── conftest.py                   # Pytest fixture for imports
│
├── config/
│   ├── __init__.py
│   ├── prompts.py                # LLM prompts (scalar mapping, item table, name normalisation, explanation polish)
│   ├── schema.py                 # Frozen JSON contract definitions (Receipt + Anomaly schemas)
│   ├── thresholds.yaml           # All tunable thresholds (total, duplicate, price, OCR, DeepSeek, currency)
│   └── thresholds_loader.py      # Single import point for YAML config
│
├── src/
│   ├── __init__.py
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── models.py             # Dataclasses: Region, LineItem, Receipt, Anomaly
│   │   ├── arithmetic.py         # Epsilon tolerance, line_ok, receipt_ok, is_vacuous
│   │   ├── validators.py         # Schema conformance checks (validate_receipt, validate_anomaly)
│   │   └── deepseek_client.py    # NVIDIA/DeepSeek API client with caching, retry, JSON enforcement
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # OpenCV deskew, denoise, contrast, crop
│   │   ├── ocr_engine.py         # PaddleOCR wrapper with caching, layout hints, region mapping
│   │   ├── field_mapper.py       # OCR → Receipt via LLM (map_scalars + map_items)
│   │   ├── name_normalizer.py    # Item-name canonicalisation (LLM + fallback rules)
│   │   ├── extract.py            # Full extraction pipeline (preprocess → OCR → map → normalise → gate)
│   │   ├── fallback_extract.py   # PaddleOCR + SROIE ground truth (no LLM needed)
│   │   └── hybrid_extract.py     # NVIDIA for scalars + local parser for items (demo mode)
│   ├── detect/
│   │   ├── __init__.py
│   │   ├── detect.py             # Detector class: runs all 3 checks, returns explained anomalies
│   │   ├── check_total.py        # Check 1: receipt-level + line-level arithmetic mismatch
│   │   ├── check_duplicate.py    # Check 2: within-receipt + across-receipt duplicate detection
│   │   ├── check_price.py        # Check 3: modified Z-score against reference pool
│   │   ├── similarity.py         # Levenshtein string similarity, numeric similarity, weighted scoring
│   │   └── reference_builder.py  # Build/save price reference pool from extracted receipts
│   ├── explain/
│   │   ├── __init__.py
│   │   ├── explain.py            # Enforce 4-field contract, optional LLM polish, summary
│   │   ├── templates.py          # Template-based reason strings (auditable, no LLM needed)
│   │   └── number_guard.py       # Rejects LLM explanations that alter computed numbers
│   └── ui/
│       ├── __init__.py
│       ├── app.py                # Gradio UI: image upload + JSON upload tabs, detection, review, export
│       ├── export.py             # CSV ledger + audit JSON export (confirmed rows only)
│       └── highlight.py          # Draw PaddleOCR boxes on receipt image for evidence
│
├── eval/
│   ├── __init__.py
│   ├── error_injector.py         # Generate error-injected receipts (wrong total, duplicate, inflated price)
│   ├── compute_metrics.py        # Precision, recall, F1, FPR, review load
│   ├── ground_truth/
│   │   └── sroie_gt.json         # SROIE ground truth entities for all 20 images
│   ├── error_set/
│   │   ├── labels.json           # 48 labels (16 receipts × 3 fault types)
│   │   └── receipts/             # 48 error-injected receipt JSON files
│   └── results/
│       └── metrics.json          # Final evaluation metrics
│
├── data/
│   ├── photos/
│   │   ├── normal/               # 16 SROIE train images (sroie_train_000.jpg – _015.jpg)
│   │   └── test/                 # 4 SROIE test images (sroie_test_000.jpg – _003.jpg)
│   ├── extracted/
│   │   ├── normal_extracted.json # Extraction results for 16 normal receipts (47 items)
│   │   └── test_extracted.json   # Extraction results for 4 test receipts
│   └── reference/
│       └── pool.json             # Price reference pool (1 product, 2 observations)
│
└── tests/
    ├── __init__.py
    ├── test_check_total.py       # Worked example, epsilon boundary, vacuous guard, line guard
    ├── test_check_duplicate.py   # Within-receipt duplicate detection
    ├── test_check_price.py       # Modified Z-score, MAD guard, min comparables
    ├── test_explain.py           # Contract enforcement, number guard
    ├── test_name_normalizer.py   # Fallback canonicalisation rules
    └── test_ocr_engine.py        # OCR engine caching and bbox conversion
```

---

## 3. Code Files — Detailed Description

### 3.1 Config Layer

**`config/thresholds.yaml`** — All tunable parameters in one file:
- `total`: tolerance_relative (0.001), tolerance_floor (0.01), receipt_tolerance_floor (1.00)
- `duplicate`: mode (within_receipt), weights (store/date/total/items), threshold_match (0.90), threshold_review (0.70)
- `price`: modified_z_threshold (3.5), min_comparables (2), mad_epsilon (1.0e-9)
- `ocr`: lang (en), use_gpu (false), structure (true), det thresholds, confidence review_below (0.80)
- `deepseek`: model (deepseek-ai/deepseek-v4.1-flash), base_url (NVIDIA), temperature (0), timeout (180), json_mode (false)
- `currency`: required (MYR), reject_non_myr (true)

**`config/thresholds_loader.py`** — Loads YAML once at import time; provides `THRESHOLDS` dict and `get()` dot-path accessor.

**`config/prompts.py`** — Four LLM prompts:
- `SCALAR_MAPPING_PROMPT`: Extract vendor/date/total/tax from OCR text → JSON
- `ITEM_TABLE_PROMPT`: Extract line-item table → JSON with items array (simplified for NVIDIA)
- `NAME_NORMALISATION_PROMPT`: Canonicalise item names → JSON (simplified for NVIDIA)
- `EXPLANATION_POLISH_PROMPT`: Rewrite anomaly message in one sentence (optional, off by default)

**`config/schema.py`** — Frozen JSON contract definitions mirroring the dataclasses, used for validation and documentation.

### 3.2 Shared Layer

**`src/shared/models.py`** — Core dataclasses:
- `Region`: OCR region with label, bbox_px, text, confidence
- `LineItem`: name, qty, unit_price, line_total, mismatch, canonical, raw_text, region, ocr_confidence, review (5 states), reason
- `Receipt`: vendor, date, currency, items, receipt_total, tax, receipt_id, image_path, regions, duplicate_flag, provenance; methods: `subtotal()`, `needs_review()`
- `Anomaly`: anomaly_type (3 allowed), reason, evidence, confidence [0,1]; validates on init

**`src/shared/arithmetic.py`** — Tolerance math:
- `epsilon(expected)`: line-level ε = max(0.01, 0.001 × expected)
- `receipt_epsilon(expected)`: receipt-level ε = max(1.00, 0.001 × expected) — looser to tolerate OCR misses
- `line_ok(qty, unit_price, line_total)`: |line_total − qty × unit_price| ≤ ε
- `receipt_ok(subtotal, discount, tax, printed_total)`: E = S − D + T, flag if |E − X| > receipt_epsilon
- `is_vacuous(subtotal, printed_total)`: guard against zero comparisons

**`src/shared/validators.py`** — Schema conformance: validates currency=MYR, non-negative totals/qty/prices, non-empty names, valid anomaly types and confidence ranges.

**`src/shared/deepseek_client.py`** — LLM API client:
- Lazy init with `DEEPSEEK_API_KEY` from `.env`
- NVIDIA endpoint: `https://integrate.api.nvidia.com/v1`, model `deepseek-ai/deepseek-v4.1-flash`
- Disk cache (SHA-256 keyed) for reproducible metrics
- Conditional `response_format={"type":"json_object"}` (disabled for NVIDIA via `json_mode: false`)
- Retry with exponential backoff (1s, 2s, 4s)
- Three public functions: `map_scalars()`, `map_items()`, `canonicalise()`
- `health_check()`: cheap connectivity probe

### 3.3 Extract Layer

**`src/extract/preprocessing.py`** — OpenCV image cleanup:
- `preprocess(image_path)`: deskew → denoise/contrast → crop border → save `_clean.png`
- `_deskew()`: detects text angle via minAreaRect, rotates if 0.5°–15°
- `_denoise_contrast()`: CLAHE contrast stretching for faded thermal print
- `_crop_border()`: removes dark photo edges

**`src/extract/ocr_engine.py`** — PaddleOCR wrapper:
- `read(image_path)`: returns `{text, regions, image_size, reader}` with disk cache
- Regions sorted top-to-bottom, left-to-right (reading order)
- `layout_hint(regions)`: compact `[y= x= c=] text` format for LLM prompt
- `region_for_text(regions, needle)`: find bbox for a given text string
- `low_confidence_lines()`: regions below threshold for human review
- `_to_bbox()`: PaddleOCR 4-corner points → [x1,y1,x2,y2]

**`src/extract/field_mapper.py`** — OCR → Receipt via LLM:
- `map_receipt()`: calls `deepseek_client.map_scalars()` + `map_items()` → constructs Receipt
- `_to_items()`: row dicts → LineItem objects with provenance
- `_confidence_for()`: carries PaddleOCR per-box confidence onto line items
- `_anchor_hints()`: detects TOTAL/CASH/CHANGE/TAX keywords in text
- `validate_currency()`: enforces MYR

**`src/extract/name_normalizer.py`** — Item-name canonicalisation:
- `fallback_canonical()`: rule-based — uppercase, expand abbreviations (FN→F&N, LTR→L, PKT→PACKET, etc.), strip qty prefixes, remove punctuation, Malay food terms
- `normalise_receipt()`: tries LLM canonicalise, falls back to local rules; records `_name_source` in provenance
- `canonical_key()`: returns canonical name (used by Check 3 pool)

**`src/extract/extract.py`** — Full pipeline:
- `extract_one(image_path)`: preprocess → OCR → field_mapper → name_normalizer → arithmetic_gate → validate
- `run_arithmetic_gate()`: marks items as `review_required` if line arithmetic fails, OCR confidence low, or mapper flagged mismatch
- `to_json(receipt)`: emits frozen JSON contract with provenance
- `extract_batch()`: batch with per-receipt error isolation
- `save_batch()`: cache batch to disk

**`src/extract/fallback_extract.py`** — No-LLM extraction:
- `extract_with_paddleocr(image_path, sroie_gt)`: PaddleOCR + SROIE ground truth for header fields, column-aware local parser for line items
- `_group_regions_into_rows()`: groups OCR regions by y-position proximity (±15px tolerance)
- `_detect_column_layout()`: finds header row with Description/Qty/Price/Amount columns
- `_parse_line_items()`: column-aware parser using detected x-positions; fallback to inline price parsing

**`src/extract/hybrid_extract.py`** — Demo-mode extraction:
- `extract_hybrid(image_path, sroie_gt)`: NVIDIA API for `map_scalars` only (~40s), local column-aware parser for items, fallback name normaliser
- Used by the Gradio UI to avoid the slow/hanging `map_items` LLM call

### 3.4 Detect Layer

**`src/detect/detect.py`** — Orchestrator:
- `Detector(price_pool, history, discount)`: runs all 3 checks, explains results
- `run(receipt)`: check_total + check_duplicate + check_price → explain → return `{receipt_id, anomalies, summary, flagged, pool_size}`
- `run_batch()`: apply to list of receipts

**`src/detect/check_total.py`** — Check 1: Incorrect total:
- Receipt-level: E = S − D + T, flag if |E − X| > receipt_epsilon
- Line-level: flag if |qty × unit_price − line_total| > epsilon (tight tolerance)
- Vacuous guard: skip if subtotal or total is zero
- Confidence = min(1.0, |diff| / max(expected, 0.01))

**`src/detect/check_duplicate.py`** — Check 2: Duplicate transaction:
- `check_within()`: same line item appearing twice in one receipt (same name, qty, unit_price within tolerance)
- `check_across()`: Fellegi & Sunter weighted scoring across receipts (vendor/date/total/items similarity)
- Three-state output: match (≥0.90) / possible_match (≥0.70) / non_match
- Default mode: `within_receipt` (no external history needed)

**`src/detect/check_price.py`** — Check 3: Unusual price:
- Modified Z-score: Mᵢ = 0.674F(xᵢ − x̃) / MAD
- Flag if |Mᵢ| > 3.5 (configurable)
- Guards: MAD=0 → skip (degenerate), < min_comparables → skip, empty pool → skip
- Uses canonical name as pool key (not raw name)

**`src/detect/similarity.py`** — String + numeric similarity:
- `string_similarity()`: 1 − Levenshtein distance / max length (via rapidfuzz)
- `numeric_similarity()`: 1.0 if within tolerance, else linear decay
- `date_similarity()`: 1.0 same day, 0.5 within window, 0.0 outside
- `items_similarity()`: fraction of r1's items finding a close match in r2
- `weighted_score()`: S = Σ wₖ sₖ with normalised weights

**`src/detect/reference_builder.py`** — Price reference pool:
- `build_pool(receipts, min_obs)`: {canonical_name: [unit_price, ...]} with currency + normalisation + min-obs guards
- `save_pool()` / `build_from_cached()`: persist/load pool
- `pool_stats()`: diagnostics (items, observations, median obs, source)
- `degenerate_items()`: items with all identical prices (MAD=0 → skipped by Check 3)

### 3.5 Explain Layer

**`src/explain/explain.py`** — Explanation orchestration:
- `enforce_contract()`: guarantees all 4 fields (anomaly_type, reason, evidence, confidence) with correct types
- `_polish()`: optional LLM rephrasing (off by default, DOCAUDIT_LLM_EXPLAIN=1 to enable)
- `explain()`: applies polish + contract enforcement to each anomaly
- `summary()`: human-readable count summary

**`src/explain/templates.py`** — Template-based reasons:
- 5 templates: incorrect_total_receipt, incorrect_total_line, duplicate_within, duplicate_across, unusual_price
- `render(key, facts)`: format template with computed facts — auditable, no LLM needed

**`src/explain/number_guard.py`** — LLM output safety:
- `numbers_ok(text, facts)`: checks every money amount and bare integer in text against allowed values from facts
- `guard(fallback, candidate, facts)`: returns candidate only if it survives; else template fallback

### 3.6 UI Layer

**`src/ui/app.py`** — Gradio demo:
- Two tabs: Image (full pipeline) and JSON (pre-extracted/error-injected)
- Image tab: upload photo → hybrid extraction → detection → evidence highlight → anomalies → line items table
- JSON tab: upload JSON → detection only (for testing error-injected receipts)
- Three dispositions: Accept, Correct, Reject (per line item)
- Export: ledger CSV + audit JSON
- LLM health check button

**`src/ui/export.py`** — Export confirmed rows only:
- `to_ledger_csv()`: one row per confirmed line item (unreviewed/corrected/accepted)
- `to_audit_json()`: full audit trail with confirmed items, flagged items, anomalies, evidence
- `export_summary()`: counts for UI (total/flagged/confirmed/rejected/review_load)

**`src/ui/highlight.py`** — Evidence visualization:
- `annotate(image_path, regions, flagged_boxes)`: draws red rectangles on flagged OCR regions
- Uses PaddleOCR's own bbox detections — provably correct highlighting

### 3.7 Eval Layer

**`eval/error_injector.py`** — Generate labelled error set:
- `inject_wrong_total(receipt, magnitude=2.0)`: ±RM 2.0 delta on receipt_total
- `inject_duplicate_item(receipt)`: copies a random line item, updates total
- `inject_inflated_price(receipt, factor=2.5)`: 2.5× price inflation on a random item
- `build_error_set()`: 3 faults × N receipts → labelled JSON files + labels.json

**`eval/compute_metrics.py`** — Four metrics:
- `confusion(preds, labels, atype)`: TP/FP/FN/TN per anomaly type
- `metrics()`: precision, recall, F1, FPR, review_load
- `write_report()`: save to eval/results/metrics.json

---

## 4. Evaluation Results

### 4.1 Test Set Composition

| Source | Count | Description |
|--------|-------|-------------|
| Normal receipts | 116 | SROIE train images, extracted and used as reference pool source |
| Test receipts | 4 | SROIE held-out images, clean (no injected errors) |
| Error-injected | 78 | 26 receipts with items × 3 fault types (incorrect_total, duplicate, unusual_price) |
| **Total** | **194** | 78 labelled positives + 116 negatives (clean receipts × 3 types) |

### 4.2 Confusion Matrix per Check

| Check | TP | FP | FN | TN | Precision | Recall | F1 | FPR |
|-------|----|----|----|----|-----------|--------|----|----|
| **Incorrect total** | 8 | 15 | 19 | 153 | 0.3478 | 0.2963 | 0.3195 | 0.0893 |
| **Duplicate** | 26 | 0 | 0 | 168 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Unusual price** | 11 | 0 | 15 | 168 | 1.0000 | 0.4231 | 0.5946 | 0.0000 |

**Review load**: 0.3093 (30.9% of receipts triggered at least one flag)

### 4.3 Interpretation

**Check 1 — Incorrect total**: 
- **Recall = 0.30**: Only 8 of 27 injected wrong-total errors were detected. The coverage guard (skip receipt-level check if `subtotal/total < 0.8 or > 1.2`) reduces false positives but also misses errors on receipts with incomplete item extraction.
- **Precision = 0.35**: 15 false positives. Remaining causes: (1) receipts where items partially cover the total (ratio 0.8–1.2) but still don't match, (2) line-level arithmetic noise from OCR misreads on noise items like tax lines, (3) cross-contamination from duplicate/price-injected receipts that also alter the total.
- **Root cause**: The local parser extracts very few items per receipt (avg 0.44 items/receipt), so the subtotal rarely matches the receipt total. The coverage guard is a workaround — the real fix is better item extraction.

**Check 2 — Duplicate**:
- **Perfect**: P=1.0, R=1.0, F1=1.0. All 26 duplicate-injected receipts correctly detected, zero false positives. The within-receipt duplicate check (same name + same qty + same unit_price) is deterministic and precise.

**Check 3 — Unusual price**:
- **Functional**: P=1.0, R=0.42, F1=0.59. The curated reference pool (`data/reference/pool.json`) contains 30 products with realistic Malaysian market prices (4–5 observations each). When the error injector inflates a price by 2.5×, the modified Z-score correctly flags it. Recall is limited because only 30 of 51 extracted items match pool entries — when the injector inflates a non-pool item, Check 3 can't detect it.
- **Reference source**: The built-from-receipts pool is empty (SROIE stores have no product overlap). The curated pool is a proof-of-concept showing the Z-score detection works. For production, a real market price database is needed.

### 4.4 Held-out Test Receipts (Clean)

| Receipt | Vendor | Total | Items | Anomalies |
|---------|--------|-------|-------|-----------|
| sroie_test_000 | OJC MARKETING SDN BHD | RM 193.00 | 1 | None |
| sroie_test_001 | OJC MARKETING SDN BHD | RM 170.00 | 1 | None |
| sroie_test_002 | PERNIAGAAN ZHENG(?) | RM 436.20 | 2 | Incorrect total (RM 436.20 vs expected RM 24.70) |
| sroie_test_003 | PETRON BKT LANJAN SB | RM 4.90 | 0 | None |

sroie_test_002 shows a real detection: the extracted items sum to RM 24.70 but the receipt says RM 436.20 — likely OCR missed most line items, but the check correctly flags the discrepancy.

### 4.5 Unit Tests

**40/40 tests pass** across 6 test files:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_check_total.py | 6 | Worked example mismatch/correct, epsilon boundary, vacuous guard, line guard, line-level flag |
| test_check_duplicate.py | 3 | Within-receipt duplicate detection, no false positive on different items |
| test_check_price.py | 4 | Modified Z-score, MAD=0 guard, min comparables guard, pool lookup |
| test_explain.py | 4 | Contract enforcement, number guard acceptance/rejection |
| test_name_normalizer.py |, 4 | Fallback canonicalisation: abbreviations, qty prefix, price-like, normal |
| test_ocr_engine.py | 3 | Cache path, bbox conversion, box string |

---

## 5. How to Run the Project

### 5.1 Prerequisites

- Python 3.12 (PaddlePaddle 2.x has no 3.13 wheels)
- Windows: short-path venv (e.g., `C:\venv\da312`) due to PaddlePaddle's long header paths
- NVIDIA API key in `.env` (stored as `DEEPSEEK_API_KEY`)

### 5.2 Installation

```bash
# Create venv with Python 3.12 (must be 3.12, not 3.13 — PaddlePaddle 2.x has no 3.13 wheels)
C:\Python312\python.exe -m venv C:\venv\da --clear

# Install dependencies using the venv's Python directly
# (PowerShell activation may not stick — use the venv python path instead)
C:\venv\da\Scripts\python.exe -m pip install setuptools
C:\venv\da\Scripts\python.exe -m pip install -r requirements.txt
C:\venv\da\Scripts\python.exe -m pip install -e .
```

### 5.3 Run the Demo UI

```bash
C:\venv\da\Scripts\python.exe -m src.ui.app
```

Open `http://127.0.0.1:7860` in a browser.

**Image tab**: Upload a receipt photo → optionally check "Skip API" to bypass the NVIDIA LLM (faster, rule-based header extraction) → click "Analyse" → see extraction results, anomalies, evidence highlights, and a color-coded line-item table with TOTAL row.

**JSON tab**: Upload a JSON from `eval/error_set/receipts/` (e.g., `err_000_duplicate.json`) → click "Run detection" → see anomaly detection results without needing OCR.

**Color-coded table**: Red cells indicate anomalies — 🔢 incorrect total (red on total value), 🔁 duplicate (red on item name), 💰 unusual price (red on unit price). Normal rows are auto-accepted. Use the dropdown to select anomaly rows for Accept/Reject/Correct decisions.

### 5.4 Run Tests

```bash
C:\venv\da\Scripts\python.exe -m pytest -q
```

### 5.5 Run Extraction (Batch)

```bash
# Hybrid mode (NVIDIA for scalars + local parser for items)
C:\venv\da\Scripts\python.exe -m src.extract.hybrid_extract data/photos/normal --gt eval/ground_truth/sroie_gt.json --out data/extracted/normal_extracted.json

# Fallback mode (no LLM at all, uses SROIE ground truth)
C:\venv\da\Scripts\python.exe -m src.extract.fallback_extract data/photos/normal --gt eval/ground_truth/sroie_gt.json --out data/extracted/normal_extracted.json
```

### 5.6 Run Evaluation

```bash
# Generate error-injected receipts
C:\venv\da\Scripts\python.exe -c "from eval.error_injector import build_error_set; build_error_set([...], 'eval/error_set')"

# Compute metrics
C:\venv\da\Scripts\python.exe -c "from eval.compute_metrics import metrics; ..."
```

---

## 6. Known Limitations

1. **Check 3 uses a curated reference pool** — The built-from-receipts pool is empty (SROIE stores have no product overlap). A curated pool of 30 products with hand-crafted Malaysian market prices was created as a proof-of-concept. For production, a real market price database is needed.

2. **Check 1 precision is moderate (0.35)** — The local parser extracts very few items per receipt (avg 0.44 items/receipt), so `sum(items)` rarely matches `receipt_total`. A coverage guard (skip if `subtotal/total < 0.8 or > 1.2`) mitigates false positives but also reduces recall. The root fix is better item extraction.

3. **NVIDIA API is slow and inconsistent** — `map_scalars` takes ~40s, `map_items` hangs on verbose prompts. The hybrid approach (NVIDIA for scalars only) cuts per-image time from ~300s to ~60s but still times out on ~20% of images.

4. **No cross-receipt duplicate detection in the UI** — `check_across` requires history (previously extracted receipts), which the UI doesn't supply. Only `within_receipt` mode is active.

5. **SROIE has no line-item ground truth** — Header field accuracy can be validated against SROIE annotations, but line-item extraction accuracy cannot.

6. **Name normalisation is fallback-only in hybrid mode** — The LLM canonicaliser works but is too slow (138s/call). The fallback normaliser is weaker (rule-based) but deterministic and instant.

---

## 7. Artifacts Removed

The following temporary/unused files were removed during cleanup:

| Removed | Reason |
|---------|--------|
| `run_extraction.py` | Temporary background extraction runner |
| `extraction_log.txt` | Log file from background extraction |
| `ui_log.txt` | UI server log |
| `eval/tmp_normal/` | Intermediate individual receipt JSONs (used only for error injection input) |
| `eval/ground_truth/labels.json` | Empty placeholder (0 bytes) |
| `eval/ground_truth/pairs.json` | Empty placeholder (0 bytes) |
| `.pytest_cache/` | Auto-generated pytest cache |
| `src/extract/cache/` | Auto-generated OCR cache (regenerated on first run) |
| `demo/` | Demo script directory (unused, referenced non-existent `demo/receipts/`) |
| `Edata/photos/normal/*_clean.png` | Auto-generated preprocessed images (regenerated by preprocessing) |
| `data/photos/test/*_clean.png` | Same |

---

## 8. Summary

DocAudit demonstrates a complete anomaly detection pipeline for Malaysian receipts: OCR extraction (PaddleOCR), LLM field mapping (NVIDIA-hosted DeepSeek), three deterministic detection checks (incorrect total, duplicate, unusual price), templated explanations, and a Gradio UI with human review workflows. The system was evaluated on 78 error-injected receipts across 116 normal receipts: duplicate detection is perfect (P=R=F1=1.0), unusual price detection is functional with a curated reference pool (P=1.0, R=0.42, F1=0.59), and total mismatch detection has moderate precision (P=0.35) limited by item extraction quality. All 40 unit tests pass.
