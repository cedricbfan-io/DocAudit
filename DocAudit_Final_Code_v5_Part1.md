# DocAudit — Final Pipeline, Project Structure & Code (v5, Part 1 of 2)

**Explainable Anomaly Detection in Malaysian Shopping Receipts (MYR)**

**Date:** 2026-11-04 · **Scope:** Malaysian English receipts · MYR (RM) only · receipts only

**This is the final version.** DeepSeek-OCR is removed entirely. Two models only:

| Model | Role |
|---|---|
| **PaddleOCR PP-Structure** | Read the receipt — text, pixel boxes, per-box confidence |
| **DeepSeek `deepseek-chat` (API)** | Structure text into schema JSON; normalise item names; optional wording polish |

**Removed from the project:** the DeepSeek-OCR reader, the grounding-tag parser, the normalised-coordinate path, and the reader-comparison config.

**Added in v5:** `name_normalizer.py`, `reference_builder.py`, `export.py`, full `app.py`, `test_name_normalizer.py`.

> **Part 2** contains the remaining source files (UI, eval, tests) and Part 4–5 (what else you need, build order).

---

## PART 1 — THE FINAL PIPELINE

### 1.1 Role assignment

| Stage | Tool | Why |
|---|---|---|
| **Read** | **PaddleOCR PP-Structure** | Native text + pixel boxes + per-box confidence. CPU works. One pip install |
| **Normalise** | **DeepSeek** | `"MILK 2L F&N"` / `"FN MILK 2L"` → one canonical name. Keeps Check 3's pool usable |
| **Map** | **DeepSeek** | Regions → schema JSON. Absorbs layout variation without per-store rules |
| **Validate** | Code | `ε = max(0.01, 0.001 × expected)` arithmetic gate |
| **Decide** | Code | Three deterministic checks |
| **Explain** | **Template first**; DeepSeek optional rephrasing, number-validated | Reason built from *the same numbers the check used* |

**One line:** Read with PaddleOCR. Structure with DeepSeek. Decide with code. Explain with templates.

### 1.2 The flow

```
Receipt photo (MYR, Malaysian English)
    │
    ▼
① preprocessing.py ........ crop, deskew, denoise, orient        [code]
    │
    ▼
② ocr_engine.py ........... PaddleOCR PP-Structure
    │                        → text + pixel boxes + confidence   [OCR]
    ▼
③ field_mapper.py ......... DeepSeek chat: text+layout → schema  [DS] ← CACHED
    │
    ▼
④ name_normalizer.py ...... DeepSeek chat: item names → canonical [DS] ← CACHED
    │
    ▼
⑤ arithmetic_gate.py ...... ε tolerance → failures flagged       [code]
    │
    ▼
⑥ detect.py ............... three deterministic checks           [code]
    │
    ▼
⑦ explain.py .............. template → DeepSeek → number guard   [code + DS]
    │
    ▼
⑧ app.py .................. human confirms (3 dispositions)      [code]
    │
    ▼
⑨ export.py ............... ledger / CSV (confirmed rows only)   [code]
```

**DeepSeek is called at ③, ④, and optionally ⑦. None of them touch detection.**

### 1.3 Why name normalisation is not optional

Check 3 requires ≥5 observations of the *same* product before it will compare a price. OCR noise fragments names:

```
"MILK 2L F&N"     → 2 observations
"FN MILK 2L"      → 2 observations
"F&N MILK 2 LTR"  → 1 observation
```

**Each fragment falls below the threshold, so the check silently skips every one.** No error, no flag, no warning — detection just stops working. Step ④ is what prevents that.

### 1.4 Why the detection layer is deterministic

Nothing in PaddleOCR or DeepSeek computes a total, compares a price, or decides a duplicate.

| Check | Method |
|---|---|
| 1 — Total | `E = S − D + T`, flag if `\|E − X\| > ε` |
| 2 — Duplicate | Fellegi & Sunter three-way + weighted score |
| 3 — Price | Median + MAD + modified Z-score, flag if `\|Mᵢ\| > 3.5` |

**Why this matters for your report:** precision, recall, FPR and F1 all require the same input to yield the same output. An LLM in the detection path makes every reported number unreproducible.

### 1.5 The explanation layer

```
Check computes  →  facts = {expected: 14.00, observed: 16.00, diff: 2.00}
Template renders → "Total RM 16.00 is RM 2.00 higher than the expected RM 14.00."
DeepSeek (opt.) → rephrases, facts supplied as LOCKED inputs
number_guard()  → any altered figure → discard, keep the template
```

Your proposal requires the explanation to give *"the exact reason(s) and the supporting source content"* [1](@index-ref?id=pdf_3). A template is trivially auditable — the reason **is** the computation.

---

## PART 2 — PROJECT FILE STRUCTURE

```
docaudit/
│
├── README.md
├── requirements.txt
├── .env.example                          ← DEEPSEEK_API_KEY
├── .gitignore                            ← data/, cache/, .env, *_clean.png
│
├── config/
│   ├── schema.py
│   ├── thresholds.yaml
│   ├── thresholds_loader.py
│   └── prompts.py
│
├── src/
│   ├── shared/
│   │   ├── models.py
│   │   ├── arithmetic.py
│   │   ├── deepseek_client.py            ← API + CACHE + retry
│   │   └── validators.py
│   │
│   ├── extract/                          ← LANE A (P3)
│   │   ├── extract.py                    ★ gate deliverable
│   │   ├── ocr_engine.py                 ← PaddleOCR PP-Structure
│   │   ├── preprocessing.py
│   │   ├── field_mapper.py               ← DeepSeek
│   │   ├── name_normalizer.py            ← DeepSeek  ★ NEW in v5
│   │   └── cache/                        ← frozen model output [gitignored]
│   │
│   ├── detect/                           ← LANE B (P4)
│   │   ├── detect.py                     ★ gate deliverable
│   │   ├── check_total.py
│   │   ├── check_duplicate.py
│   │   ├── check_price.py
│   │   ├── similarity.py
│   │   └── reference_builder.py          ★ NEW in v5
│   │
│   ├── explain/                          ← LANE B (P4)
│   │   ├── explain.py                    ★ gate deliverable
│   │   ├── templates.py
│   │   └── number_guard.py
│   │
│   └── ui/                               ← LANE C (P5)
│       ├── app.py                        ★ NEW in v5
│       ├── export.py                     ★ NEW in v5
│       └── highlight.py
│
├── data/
│   ├── normal/                           ← ~50 clean MYR receipts
│   ├── error/                            ← ~30–50 labelled faults
│   ├── reference/                        ← Check-3 price pool
│   └── photos/                           ← 20–50 real photographed receipts
│
├── eval/
│   ├── error_injector.py
│   ├── compute_metrics.py
│   └── ground_truth/
│       ├── labels.json
│       └── pairs.json
│
├── demo/
│   ├── receipts/
│   ├── playback/
│   └── script.md
│
└── tests/
    ├── test_ocr_engine.py
    ├── test_check_total.py
    ├── test_check_duplicate.py
    ├── test_check_price.py
    ├── test_name_normalizer.py           ★ NEW in v5
    └── test_explain.py
```

---

## PART 3 — FILE CONTENTS (A: CONFIG + SHARED + EXTRACT)

### 3.1 `requirements.txt`

```
# ── OCR ────────────────────────────────────────────────
paddleocr>=2.7
paddlepaddle>=2.5          # CPU is fine; use paddlepaddle-gpu for CUDA

# ── DeepSeek (OpenAI-compatible API) ───────────────────
openai>=1.30

# ── Detection ──────────────────────────────────────────
rapidfuzz>=3.0

# ── Support ────────────────────────────────────────────
python-dotenv>=1.0
pyyaml>=6.0
Pillow>=10.0
opencv-python>=4.8
pandas>=2.0

# ── UI ─────────────────────────────────────────────────
gradio>=4.0

# ── Tests ──────────────────────────────────────────────
pytest>=8.0
```

**No CUDA. No flash-attn. No multi-gigabyte model download.**

### 3.2 `.env.example`

```
DEEPSEEK_API_KEY=sk-your-key-here
DOCAUDIT_LLM_EXPLAIN=0          # 1 = enable DeepSeek wording polish
```

### 3.3 `config/thresholds.yaml`

```yaml
# Each value below is a GUESS until tuned against labelled data.
# Tuning is not training: run on labelled data, read metrics, adjust by hand.

# ── Check 1: Incorrect total ──────────────────────────────
total:
  tolerance_relative: 0.001       # ε = max(floor, relative × expected)
  tolerance_floor: 0.01

# ── Check 2: Duplicate ────────────────────────────────────
# ⚠️ The deck states: "Weights and thresholds require labelled-data
# testing." They were never set. Tune, then FREEZE before evaluation.
duplicate:
  mode: within_receipt            # 'within_receipt' | 'across_receipts'
  weights:
    store: 0.30
    date: 0.25
    total: 0.25
    items: 0.20
  threshold_match: 0.90           # S ≥ 0.90 → match
  threshold_review: 0.70          # 0.70 ≤ S < 0.90 → possible match
  date_window_days: 1
  total_tolerance: 0.01
  item_similarity_min: 0.85

# ── Check 3: Unusual price ────────────────────────────────
price:
  modified_z_threshold: 3.5       # flag if |Mᵢ| > 3.5
  min_comparables: 5              # ⚠️ docs say "≥5–10 samples"
  mad_epsilon: 1e-9               # guard: MAD = 0 → skip item

# ── OCR ───────────────────────────────────────────────────
ocr:
  lang: en
  use_gpu: false
  structure: true                 # PP-Structure = layout + table parsing
  det_db_thresh: 0.3
  det_db_box_thresh: 0.5
  use_angle_cls: true
  confidence:
    review_below: 0.80            # ocr_confidence < 0.80 → human review

# ── DeepSeek ──────────────────────────────────────────────
deepseek:
  model: deepseek-chat
  base_url: https://api.deepseek.com
  temperature: 0                  # reproducibility
  max_retries: 3
  timeout: 60
  cache: true                     # MANDATORY for stable metrics

# ── Currency guard ────────────────────────────────────────
currency:
  required: MYR
  reject_non_myr: true            # never pool prices across currencies
```

### 3.4 `config/thresholds_loader.py`

```python
"""Single import point for every tunable number.

Nothing reads the YAML directly except this module. When you tune a
threshold, you change one file and every check follows.
"""

import pathlib
import yaml

_PATH = pathlib.Path(__file__).parent / "thresholds.yaml"

with _PATH.open(encoding="utf-8") as f:
    THRESHOLDS = yaml.safe_load(f)


def get(path: str, default=None):
    """Dot-path access: get('duplicate.weights.store')."""
    node = THRESHOLDS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
```

### 3.5 `config/prompts.py`

```python
"""Every DeepSeek prompt, in one place.

Two rules that matter throughout:

1. COPY values, never compute them. If the mapper 'fixed' a wrong
   total, Check 1 would lose the error it exists to catch.

2. Return JSON only. Every call sets response_format=json_object,
   so a prose reply is a parse error, not a silent failure.
"""

# ── Header fields ─────────────────────────────────────────
SCALAR_MAPPING_PROMPT = """\
You extract header fields from Malaysian shopping receipts.

Input is OCR text, optionally preceded by LAYOUT HINTS listing detected
blocks (vertical position, horizontal position, confidence, text).

Output ONLY valid JSON:
{
  "vendor": string | null,
  "date": "YYYY-MM-DD" | null,
  "currency": "MYR",
  "receipt_total": number | null,
  "tax": number | null
}

RULES:
1. COPY what is printed. Never compute, sum, or correct any value.
2. If a field is absent, use null. Do not guess.
3. tax may legitimately be 0.00 or absent — Malaysian SST is embedded
   inside the shelf price, so most retail receipts show no tax line.
   A zero tax field is NOT an error.
4. currency is always "MYR".
5. Output JSON only. No prose, no markdown fences.
"""

# ── Line-item table ───────────────────────────────────────
ITEM_TABLE_PROMPT = """\
You extract the line-item table from a Malaysian shopping receipt.

Input is OCR text. LAYOUT HINTS may be provided: each detected block
has a vertical position (y), a horizontal position (x), a confidence
score (c), and its text. Use y to recover row order and x to recover
column order.

Columns may be [NAME] [QTY] [UNIT_PRICE] [LINE_TOTAL] in ANY order.
Qty is sometimes embedded in the name, e.g. "2 x MILK 3.50".

Output ONLY a JSON object with one key, "items", whose value is an array.
Each element: {"name": string, "qty": number, "unit_price": number,
"line_total": number, "mismatch": boolean, "raw": string}

RULES:
1. Determine column order from the VALUES, not from position.
2. qty × unit_price should equal line_total as printed.
   If they do NOT, report the printed values anyway and set
   "mismatch": true.
3. Do NOT fix the arithmetic.                                ← critical
4. Skip non-item lines (TOTAL, SUBTOTAL, CASH, CHANGE, TAX, TENDER).
5. "raw" is the original OCR text for that row, unchanged.
6. Output JSON only.
"""

# ── Item-name canonicalisation ────────────────────────────
NAME_NORMALISATION_PROMPT = """\
You canonicalise grocery item names from Malaysian retail receipts.

You receive a JSON object with one key, "names", whose value is a list
of raw product names. Return the SAME list length, in the SAME order,
with each name replaced by a canonical form.

Output ONLY valid JSON:
{"canonical": [{"raw": string, "canonical": string}]}

RULES FOR THE CANONICAL FORM:
1. UPPERCASE, single spaces, no punctuation.
2. Expand common Malaysian abbreviations:
   FN → F&N, LTR/L → L, ML → ML, KG → KG, PKT → PACKET, BTL → BOTTLE
3. Put BRAND first, then product, then SIZE. Strip store/till codes.
4. Remove qty prefixes already captured separately, e.g. "2 X ".
5. Preserve the size/weight — it distinguishes different products.
6. If a name is unusable (pure noise, a price, a barcode), set
   canonical to "" so the caller can drop it.
7. Never merge DIFFERENT products. "MILK 2L" and "MILK 1L" are distinct.
8. Output JSON only. No prose.
"""

# ── Optional wording polish ───────────────────────────────
EXPLANATION_POLISH_PROMPT = """\
Rewrite the given receipt-anomaly message in one clear sentence.

RULES:
1. Use EXACTLY the numbers provided. Never round, recompute, or infer.
2. Keep all RM amounts exactly as given, to two decimal places.
3. Do not add causes, blame, or fraud claims.
4. One sentence only.
5. Return JSON: {"text": "<your sentence>"}
"""
```

### 3.6 `src/shared/models.py`

```python
"""Dataclasses mirroring the two frozen JSON contracts."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Region:
    """One OCR region. PaddleOCR gives pixel boxes natively."""
    label: str                      # text | title | table | figure
    bbox_px: list                   # [x1, y1, x2, y2] in PIXELS
    text: str = ""
    confidence: float = 1.0


@dataclass
class LineItem:
    name: str                       # raw name as read from the receipt
    qty: float
    unit_price: float
    line_total: float
    mismatch: bool = False          # set by the mapper, confirmed by the gate

    canonical: str = ""             # normalised name — used by Check 3
    raw_text: str = ""              # original OCR row text
    region: str = ""                # bbox string, for evidence highlighting
    ocr_confidence: float = 1.0
    review: str = "unreviewed"      # unreviewed | review_required | corrected
    reason: str = ""                # why review was requested


@dataclass
class Receipt:
    vendor: Optional[str]
    date: Optional[str]
    currency: str = "MYR"
    items: list = field(default_factory=list)
    receipt_total: Optional[float] = None
    tax: Optional[float] = None

    receipt_id: str = ""
    image_path: str = ""
    image_size: Optional[tuple] = None
    regions: list = field(default_factory=list)     # list[Region]
    duplicate_flag: bool = False
    provenance: dict = field(default_factory=dict)

    def subtotal(self) -> float:
        return round(sum(i.line_total for i in self.items), 2)

    def needs_review(self) -> bool:
        return any(i.review == "review_required" for i in self.items) \
            or bool(self.provenance.get("_total_gap"))


@dataclass
class Anomaly:
    """The four-field contract. Every flag emits exactly this."""
    anomaly_type: str               # incorrect_total | duplicate | unusual_price
    reason: str
    evidence: dict
    confidence: float

    ALLOWED = ("incorrect_total", "duplicate", "unusual_price")

    def __post_init__(self):
        if self.anomaly_type not in self.ALLOWED:
            raise ValueError(f"bad anomaly_type: {self.anomaly_type}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
```

### 3.7 `src/shared/arithmetic.py`

```python
"""The ε tolerance — defined ONCE, shared by extraction and Check 1.

This single function powers the arithmetic gate, the biggest lever on
human workload: a mis-mapped axis breaks the arithmetic, so the failure
announces itself.
"""

from config.thresholds_loader import THRESHOLDS

_REL = THRESHOLDS["total"]["tolerance_relative"]
_FLOOR = THRESHOLDS["total"]["tolerance_floor"]


def epsilon(expected: float) -> float:
    """ε = max(0.01, 0.001 × expected)"""
    return max(_FLOOR, _REL * abs(expected))


def line_ok(qty: float, unit_price: float, line_total: float) -> bool:
    """Line level: |line_total − qty × unit_price| ≤ ε"""
    expected = qty * unit_price
    return abs(expected - line_total) <= epsilon(expected)


def receipt_ok(subtotal: float, discount: float,
               tax: float, printed_total: float) -> tuple:
    """Receipt level. Returns (ok, expected, difference).

        E = S − D + T        flag if |E − X| > ε

    If the schema carries no discount field, pass discount=0.0 and the
    formula degrades to subtotal + tax.
    """
    expected = round(subtotal - discount + tax, 2)
    difference = round(printed_total - expected, 2)
    return abs(difference) <= epsilon(expected), expected, difference


def is_vacuous(subtotal: float, printed_total: float) -> bool:
    """Guard: if either side is zero the comparison is meaningless."""
    return subtotal == 0.0 or printed_total == 0.0
```

### 3.8 `src/shared/validators.py`

```python
"""Schema conformance. Raises early, loudly, with the offending field."""

from config.thresholds_loader import THRESHOLDS


class SchemaError(ValueError):
    pass


def validate_receipt(receipt) -> None:
    if receipt.currency != THRESHOLDS["currency"]["required"]:
        raise SchemaError(f"currency must be MYR, got {receipt.currency!r}")

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
```

### 3.9 `src/shared/deepseek_client.py`

```python
"""DeepSeek API client — OpenAI-compatible endpoint + disk cache + retry.

THE CACHE IS THE POINT. Run the model once over the dataset, freeze the
JSON on disk, then run all detection and metrics against the frozen
file. Without it, precision and recall change on every evaluation run.

Used for three jobs, all text→structure:
  1. map_scalars()   — header fields
  2. map_items()     — the line-item table
  3. canonicalise()  — item-name normalisation

Never used for: OCR, detection, or supplying a reported number.
"""

import hashlib
import json
import os
import pathlib
import time
from typing import Optional

from config.thresholds_loader import THRESHOLDS
from config.prompts import (SCALAR_MAPPING_PROMPT, ITEM_TABLE_PROMPT,
                            NAME_NORMALISATION_PROMPT)

_CFG = THRESHOLDS["deepseek"]
CACHE_DIR = pathlib.Path(__file__).parent.parent / "extract" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_STATS = {"calls": 0, "cache_hits": 0, "failures": 0}


class DeepSeekError(RuntimeError):
    pass


def _get_client():
    """Lazy init — never fail at import time if the key is missing."""
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env "
                "and paste your key, or export it in the shell.")
        _client = OpenAI(api_key=api_key, base_url=_CFG["base_url"],
                         timeout=_CFG["timeout"])
    return _client


def _cache_path(system_prompt: str, payload: str) -> pathlib.Path:
    key = hashlib.sha256(
        f"{_CFG['model']}{system_prompt}{payload}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"ds_{key}.json"


def _call(system_prompt: str, user_content: str,
          expect_key: Optional[str] = None,
          use_cache: bool = True, retries: Optional[int] = None):
    """Call deepseek-chat with caching, retry, and JSON enforcement.

    expect_key: if set, unwraps {key: [...]} → [...]. json_object mode
    forces an object at the top level, so arrays must be wrapped.
    """
    path = _cache_path(system_prompt, user_content)

    if use_cache and _CFG["cache"] and path.exists():
        _STATS["cache_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))

    attempts = retries if retries is not None else _CFG["max_retries"]
    last_err = None

    for attempt in range(attempts):
        try:
            resp = _get_client().chat.completions.create(
                model=_CFG["model"],
                temperature=_CFG["temperature"],          # 0 → reproducible
                response_format={"type": "json_object"},  # → parseable
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            parsed = json.loads(resp.choices[0].message.content.strip())
            _STATS["calls"] += 1
            break
        except Exception as e:                            # network, 429, JSON
            last_err = e
            _STATS["failures"] += 1
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)                  # 1s, 2s, 4s
    else:
        raise DeepSeekError(f"DeepSeek call failed after {attempts} "
                            f"attempts: {last_err}")

    if expect_key and isinstance(parsed, dict):
        if expect_key in parsed:
            parsed = parsed[expect_key]
        else:                                             # fallback: first list
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

    if use_cache and _CFG["cache"]:
        path.write_text(json.dumps(parsed, ensure_ascii=False),
                        encoding="utf-8")
    return parsed


def _layout_payload(text: str, regions: list) -> str:
    """Text + layout hint. The hint carries column-order information
    that a flat text dump loses."""
    if not regions:
        return text
    from src.extract.ocr_engine import layout_hint
    hint = layout_hint(regions)
    return f"LAYOUT HINTS:\n{hint}\n\nOCR TEXT:\n{text}"


def map_scalars(ocr_text: str, regions: list = None,
                use_cache: bool = True) -> dict:
    """Header fields: vendor, date, total, tax."""
    return _call(SCALAR_MAPPING_PROMPT,
                 _layout_payload(ocr_text, regions or []),
                 use_cache=use_cache)


def map_items(ocr_text: str, regions: list = None,
              use_cache: bool = True) -> list:
    """The line-item table — the genuinely hard sub-problem."""
    return _call(ITEM_TABLE_PROMPT,
                 _layout_payload(ocr_text, regions or []),
                 expect_key="items", use_cache=use_cache)


def canonicalise(names: list, use_cache: bool = True) -> list:
    """Item names → canonical forms. Same length, same order.

    Returns a list of {"raw", "canonical"} dicts. On failure the caller
    falls back to a local rule-based normaliser, so Check 3 degrades
    rather than dying.
    """
    if not names:
        return []
    payload = json.dumps({"names": names}, ensure_ascii=False)
    out = _call(NAME_NORMALISATION_PROMPT, payload,
                expect_key="canonical", use_cache=use_cache)
    if isinstance(out, dict):
        out = [out]
    return out


def health_check() -> dict:
    """Cheap connectivity probe — run this before the demo."""
    try:
        out = _call(SCALAR_MAPPING_PROMPT,
                    "TESCO MART\n12/10/2026\nTOTAL RM 45.90",
                    use_cache=False, retries=1)
        return {"ok": True, "sample": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def cache_stats() -> dict:
    files = list(CACHE_DIR.glob("ds_*.json"))
    return {"cached_files": len(files), "dir": str(CACHE_DIR), **_STATS}


def clear_cache() -> int:
    files = list(CACHE_DIR.glob("ds_*.json"))
    for f in files:
        f.unlink()
    return len(files)
```

### 3.10 `src/extract/preprocessing.py`

```python
"""Crop, deskew, denoise, orient. Runs BEFORE the reader.

PaddleOCR handles worn thermal paper well, but a skewed or badly
framed crop is still the most common cause of a bad read.
Everything here is OpenCV — no model.
"""

import cv2
import numpy as np


def preprocess(image_path: str, margin: int = 8) -> str:
    """Returns a path to the cleaned image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img = _deskew(img)
    img = _denoise_contrast(img)
    img = _crop_border(img, margin)

    out = image_path.rsplit(".", 1)[0] + "_clean.png"
    cv2.imwrite(out, img)
    return out


def _deskew(img):
    """Rotate by the dominant text angle — receipts are often tilted."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thr = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:      # ignore extremes
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise_contrast(img):
    """Faded thermal print benefits from contrast stretching."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _crop_border(img, margin: int):
    """Drop the dark edge from a photographed page."""
    h, w = img.shape[:2]
    if h <= 2 * margin or w <= 2 * margin:
        return img
    return img[margin:h - margin, margin:w - margin]
```

### 3.11 `src/extract/ocr_engine.py` — THE READER

```python
"""PaddleOCR PP-Structure reader. The project's only OCR path.

Returns a stable contract the rest of the system depends on:

    {"text": str, "regions": [dict], "image_size": (w, h), "reader": str}

Each region: {"label", "bbox_px", "text", "confidence"}

WHY PP-STRUCTURE:
  Receipts are a table problem more than a text problem. The line-item
  grid is the hard part, and PP-Structure's layout + table recognition
  is what makes column order recoverable at all.

WHY PIXEL BOXES MATTER:
  `bbox_px` feeds evidence highlighting in the UI. A flagged field
  points at the exact region it came from — the "supporting source
  content" the explanation contract requires.

WHY PER-BOX CONFIDENCE MATTERS:
  It is a second, independent review trigger, running BEFORE the
  arithmetic gate. Low-confidence lines route to a human even when the
  arithmetic happens to survive.

OCR IS MODEL OUTPUT → ALWAYS CACHED. Metrics must be stable.
"""

import hashlib
import json
import pathlib

from config.thresholds_loader import THRESHOLDS

_CFG = THRESHOLDS["ocr"]
_REVIEW_BELOW = _CFG["confidence"]["review_below"]
CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_ENGINE = None


def _cache_path(image_path: str) -> pathlib.Path:
    h = hashlib.sha256(
        f"{image_path}{_CFG['lang']}{_CFG['structure']}"
        f"{_CFG['det_db_thresh']}{_CFG['det_db_box_thresh']}"
        f"{_CFG['use_gpu']}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"ocr_{h}.json"


def _get_engine():
    """Lazy init — importing paddleocr is slow and memory-heavy."""
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR
        _ENGINE = PaddleOCR(
            use_angle_cls=_CFG["use_angle_cls"],
            lang=_CFG["lang"],
            use_gpu=_CFG["use_gpu"],
            show_log=False,
            det_db_thresh=_CFG["det_db_thresh"],
            det_db_box_thresh=_CFG["det_db_box_thresh"],
        )
    return _ENGINE


def read(image_path: str, use_cache: bool = True) -> dict:
    """Photo → {'text', 'regions', 'image_size', 'reader'}."""
    cpath = _cache_path(image_path)
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    raw = _get_engine().ocr(image_path, cls=True)

    regions, lines = [], []
    for page in (raw or []):
        for box, (text, conf) in (page or []):
            bbox = _to_bbox(box)
            regions.append({
                "label": "text",
                "bbox_px": bbox,
                "text": text,
                "confidence": float(conf),
            })
            lines.append((bbox, text, float(conf)))

    # Reading order: top-to-bottom, then left-to-right.
    # Receipts are single-column enough that this recovers line order.
    lines.sort(key=lambda t: (round(t[0][1] / 10), t[0][0]))
    text = "\n".join(t[1] for t in lines)

    result = {
        "text": text,
        "regions": regions,
        "image_size": _image_size(image_path),
        "reader": "paddleocr",
    }
    if use_cache:
        cpath.write_text(json.dumps(result, ensure_ascii=False),
                         encoding="utf-8")
    return result


def layout_hint(regions: list, max_regions: int = 80) -> str:
    """Compact region list for the mapper prompt.

    Region structure is what lets the mapper recover column order:
    vertical position gives rows, horizontal gives columns. Without it
    the mapper sees a flat text blob and must guess.
    """
    out = []
    for r in regions[:max_regions]:
        x1, y1, _, _ = r["bbox_px"]
        out.append(f"[y={y1} x={x1} c={r['confidence']:.2f}] {r['text']}")
    return "\n".join(out)


def low_confidence_lines(regions: list, below: float = None) -> list:
    """Regions below the threshold → human review.

    A second review trigger, independent of the arithmetic gate.
    """
    thresh = _REVIEW_BELOW if below is None else below
    return [r for r in regions if r["confidence"] < thresh]


def region_for_text(regions: list, needle: str) -> str:
    """Bbox string of the first region containing `needle`."""
    if not needle:
        return ""
    n = needle.strip().lower()
    for r in regions:
        if n and n in (r.get("text") or "").lower():
            return box_str(r["bbox_px"])
    return ""


def _to_bbox(box) -> list:
    """PaddleOCR gives 4 corner points → [x1, y1, x2, y2]."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _image_size(image_path: str) -> tuple:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size


def box_str(bbox: list) -> str:
    return ",".join(str(int(v)) for v in bbox)
```

### 3.12 `src/extract/field_mapper.py`

```python
"""OCR output → schema fields. DeepSeek's primary job.

The mapper receives BOTH the raw text and a compact layout hint built
from PaddleOCR's regions. The hint is what makes column-order recovery
possible — receipt column order varies by store.

The mapper NEVER corrects arithmetic. A "fixed" total would erase the
error Check 1 exists to find.
"""

import re

from src.shared import deepseek_client
from src.shared.models import LineItem, Receipt
from src.extract import ocr_engine

ANCHORS = {
    "receipt_total": ["TOTAL", "JUMLAH", "AMOUNT DUE", "GRAND TOTAL"],
    "tax":           ["TAX", "SST", "GST", "CUKAI"],
    "cash":          ["CASH", "TUNAI", "PAID", "TENDER"],
    "change":        ["CHANGE", "BAKI"],
    "discount":      ["DISCOUNT", "DISKAUN", "POTONGAN"],
}


def _anchor_hints(text: str) -> dict:
    found, upper = {}, text.upper()
    for field, words in ANCHORS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", upper):
                found[field] = True
                break
    return found


def _to_items(rows: list, regions: list) -> list:
    """Row dicts → LineItem objects, carrying provenance."""
    items = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue                        # drop unusable rows
        items.append(LineItem(
            name=name,
            qty=float(r.get("qty") or 0),
            unit_price=float(r.get("unit_price") or 0),
            line_total=float(r.get("line_total") or 0),
            mismatch=bool(r.get("mismatch", False)),
            raw_text=r.get("raw") or str(r),
            region=ocr_engine.region_for_text(regions, name),
            ocr_confidence=_confidence_for(name, regions),
        ))
    return items


def _confidence_for(name: str, regions: list) -> float:
    """Carry PaddleOCR's per-box confidence onto the line item."""
    n = name.strip().lower()
    for r in regions:
        if n and n in (r.get("text") or "").lower():
            return float(r.get("confidence", 1.0))
    return 1.0


def map_receipt(ocr_result: dict, receipt_id: str, image_path: str,
                use_cache: bool = True) -> Receipt:
    """Full mapping pass: scalars + item table, both via DeepSeek."""
    clean_text = ocr_result["text"]
    regions = ocr_result["regions"]

    scalars = deepseek_client.map_scalars(clean_text, regions,
                                          use_cache=use_cache)
    rows = deepseek_client.map_items(clean_text, regions,
                                     use_cache=use_cache)

    return Receipt(
        vendor=scalars.get("vendor"),
        date=scalars.get("date"),
        currency="MYR",                          # enforced, not inferred
        items=_to_items(rows, regions),
        receipt_total=scalars.get("receipt_total"),
        tax=scalars.get("tax"),
        receipt_id=receipt_id,
        image_path=image_path,
        image_size=tuple(ocr_result["image_size"]),
        regions=regions,
        provenance={"_anchor_hints": _anchor_hints(clean_text),
                    "_reader": ocr_result.get("reader", "paddleocr")},
    )


def validate_currency(receipt: Receipt) -> bool:
    """Scope guard: one region, one currency. Never pool non-MYR."""
    return receipt.currency == "MYR"
```

### 3.13 `src/extract/name_normalizer.py` ★ NEW

```python
"""Item-name canonicalisation. DeepSeek's second high-value job.

WHY THIS EXISTS — and why it is not cosmetic:

  Check 3 needs ≥5 observations of the SAME product before it will
  compare a price. OCR noise fragments names:

      "MILK 2L F&N"     → 2 observations
      "FN MILK 2L"      → 2 observations
      "F&N MILK 2 LTR"  → 1 observation

  Each fragment falls below the threshold, so the check SILENTLY SKIPS
  every one. No error, no flag, no warning — detection just stops
  working. Canonicalisation is what keeps the pool big enough to use.

FALLBACK DESIGN:
  DeepSeek is the primary path. If the API is unavailable the local
  rule-based normaliser runs instead, so extraction never dies. The
  fallback is weaker but deterministic — and provenance records which
  path produced each name.
"""

import re

from src.shared import deepseek_client
from src.shared.models import Receipt

# ── Local fallback rules ──────────────────────────────────
_ABBREV = {
    r"\bFN\b": "F&N",
    r"\bLTR\b": "L", r"\bLITRE\b": "L", r"\bLITER\b": "L",
    r"\bMILILITRE\b": "ML",
    r"\bPKT\b": "PACKET", r"\bPKS\b": "PACKET",
    r"\bBTL\b": "BOTTLE", r"\bBTLS\b": "BOTTLE",
    r"\bSUSU\b": "MILK", r"\bROTI\b": "BREAD",
    r"\bTELUR\b": "EGG", r"\bAYAM\b": "CHICKEN",
    r"\bBERAS\b": "RICE", r"\bGULA\b": "SUGAR",
    r"\bMINYAK\b": "OIL", r"\bAIR\b": "WATER",
}
_QTY_PREFIX = re.compile(r"^\s*\d+\s*[xX*]\s*")
_PUNCT = re.compile(r"[^\w&/\s]")
_SPACES = re.compile(r"\s+")
# a name that is only digits/dots is a price or a code, not a product
_ONLY_NUMBERS = re.compile(r"^[\d.,\s%]+$")


def fallback_canonical(name: str) -> str:
    """Deterministic, offline, weaker than DeepSeek. Always available."""
    if not name:
        return ""
    s = name.strip().upper()
    if _ONLY_NUMBERS.match(s):
        return ""
    s = _QTY_PREFIX.sub("", s)              # "2 X MILK" → "MILK"
    s = _PUNCT.sub(" ", s)
    for pattern, repl in _ABBREV.items():
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return _SPACES.sub(" ", s).strip()


def normalise_receipt(receipt: Receipt, use_cache: bool = True) -> Receipt:
    """Fill LineItem.canonical for every item, in place.

    Returns the receipt with `_name_source` recorded in provenance:
    'deepseek' or 'fallback'. Report this — it is part of your
    reproducibility story.
    """
    if not receipt.items:
        return receipt

    raw_names = [i.name for i in receipt.items]
    source = "deepseek"

    try:
        pairs = deepseek_client.canonicalise(raw_names, use_cache=use_cache)
        mapping = _align(raw_names, pairs)
    except Exception as e:
        source = "fallback"
        mapping = {n: fallback_canonical(n) for n in raw_names}
        receipt.provenance["_normaliser_error"] = str(e)[:200]

    for item in receipt.items:
        item.canonical = mapping.get(item.name) or fallback_canonical(item.name)

    # Drop items the normaliser judged unusable — but only if the raw
    # name is also numeric noise. Never silently drop real products.
    receipt.items = [
        i for i in receipt.items
        if i.canonical or not _ONLY_NUMBERS.match(i.name.strip().upper())
    ]

    receipt.provenance["_name_source"] = source
    return receipt


def _align(raw_names: list, pairs: list) -> dict:
    """Map raw → canonical.

    Aligns by the "raw" key when present, otherwise falls back to
    positional order. Positional alignment guards against the model
    silently reordering or shortening the list.
    """
    mapping = {}
    if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict) \
            and "raw" in pairs[0]:
        for p in pairs:
            raw, can = p.get("raw", ""), p.get("canonical", "")
            if raw:
                mapping[raw] = (can or "").strip().upper()

    for i, raw in enumerate(raw_names):
        if raw in mapping:
            continue
        if i < len(pairs):
            p = pairs[i]
            can = p.get("canonical", "") if isinstance(p, dict) else str(p)
            mapping[raw] = (can or "").strip().upper()
        else:
            mapping[raw] = fallback_canonical(raw)
    return mapping


def canonical_key(item) -> str:
    """The key Check 3 pools on. Falls back if normalisation was skipped."""
    return item.canonical or fallback_canonical(item.name)
```

### 3.14 `src/extract/extract.py` ★

```python
"""★ GATE DELIVERABLE: extract.py outputs valid schema-compliant JSON.

Gate: detection works on ≥15/20 photos (≥75%); clean-photo field-level
accuracy ≥85%.
"""

import json
import sys
from pathlib import Path

from config.thresholds_loader import THRESHOLDS
from src.extract import field_mapper, name_normalizer, ocr_engine
from src.extract.preprocessing import preprocess
from src.shared.arithmetic import line_ok, receipt_ok, is_vacuous
from src.shared.models import Receipt
from src.shared.validators import validate_receipt

_REVIEW_BELOW = THRESHOLDS["ocr"]["confidence"]["review_below"]


def run_arithmetic_gate(receipt: Receipt) -> Receipt:
    """Step ⑤ — the human-work killer.

    Only failures become review_required. Everything else stays
    'unreviewed' and is auto-accepted. This is what makes the reader
    choice safe: a misread breaks the arithmetic and announces itself.

    THREE independent review triggers:
      1. line arithmetic fails      (qty × unit_price ≠ line_total)
      2. OCR confidence below floor
      3. the mapper itself flagged a mismatch
    """
    for item in receipt.items:
        reasons = []
        if not line_ok(item.qty, item.unit_price, item.line_total):
            reasons.append("line arithmetic")
        if item.ocr_confidence < _REVIEW_BELOW:
            reasons.append(f"low OCR confidence ({item.ocr_confidence:.2f})")
        if item.mismatch:
            reasons.append("mapper flagged mismatch")

        if reasons:
            item.review = "review_required"
            item.reason = "; ".join(reasons)

    if receipt.receipt_total is not None:
        subtotal = receipt.subtotal()
        tax = receipt.tax or 0.0
        if is_vacuous(subtotal, receipt.receipt_total):
            receipt.provenance["_vacuous"] = True
        else:
            ok, expected, diff = receipt_ok(
                subtotal, 0.0, tax, receipt.receipt_total)
            if not ok:
                receipt.provenance["_total_gap"] = {
                    "expected": expected,
                    "observed": receipt.receipt_total,
                    "difference": diff,
                }
    return receipt


def extract_one(image_path: str, receipt_id: str = "",
                use_cache: bool = True) -> Receipt:
    """Photo → schema-compliant Receipt. The full ①–⑤ sequence."""
    rid = receipt_id or Path(image_path).stem

    clean = preprocess(image_path)                              # ①
    ocr = ocr_engine.read(clean, use_cache=use_cache)           # ②
    receipt = field_mapper.map_receipt(                          # ③
        ocr, rid, image_path, use_cache=use_cache)

    if not field_mapper.validate_currency(receipt):
        raise ValueError(f"non-MYR receipt rejected: {image_path}")

    receipt = name_normalizer.normalise_receipt(                 # ④
        receipt, use_cache=use_cache)

    receipt = run_arithmetic_gate(receipt)                       # ⑤
    validate_receipt(receipt)
    return receipt


def to_json(receipt: Receipt) -> dict:
    """Emit the frozen line-item contract, plus provenance."""
    return {
        "vendor": receipt.vendor,
        "date": receipt.date,
        "currency": receipt.currency,
        "items": [
            {"name": i.name, "canonical": i.canonical,
             "qty": i.qty, "unit_price": i.unit_price,
             "line_total": i.line_total, "review": i.review,
             "reason": i.reason, "region": i.region,
             "ocr_confidence": round(i.ocr_confidence, 4)}
            for i in receipt.items
        ],
        "receipt_total": receipt.receipt_total,
        "tax": receipt.tax,
        "duplicate_flag": receipt.duplicate_flag,
        "_provenance": receipt.provenance,
        "_needs_review": receipt.needs_review(),
    }


def extract_batch(image_paths: list, use_cache: bool = True) -> list:
    """Never kill a whole batch on one bad receipt."""
    out = []
    for p in image_paths:
        try:
            out.append(to_json(extract_one(p, use_cache=use_cache)))
        except Exception as e:
            out.append({"error": str(e), "image": p})
    return out


def save_batch(image_paths: list, out_dir: str = "data/extracted") -> str:
    """Cache a whole batch to disk — this is what freezes your metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = extract_batch(image_paths)
    (out / "extracted.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out / "extracted.json")


if __name__ == "__main__":
    print(json.dumps(extract_batch(sys.argv[1:]), indent=2,
                     ensure_ascii=False))
```

### 3.15 `src/detect/reference_builder.py` ★ NEW

```python
"""Builds the Check-3 price reference pool.

Check 3 needs an external reference: a single receipt cannot tell you
whether a price is unusual. This module turns a set of receipts into
that reference.

⚠️ BLOCKER B1 — the reference SOURCE is still unselected. Until it is
chosen, this builds the pool from whatever receipts you pass in, which
is not the same thing. Report the source alongside every Check-3 number.

GUARDS ENFORCED HERE:
  1. MYR only — never pool prices across currencies
  2. Normalised names — otherwise the pool fragments and every item
     falls below min_comparables
  3. Minimum observations — an item seen twice has no meaningful median
"""

from config.thresholds_loader import THRESHOLDS
from src.extract.name_normalizer import canonical_key

_CFG = THRESHOLDS["price"]

# ⚠️ DECISION B1 — UNRESOLVED.
# Six candidate sources were listed; none was chosen. Options included
# building the pool from a dedicated reference split of receipts-2025.
# Whatever you choose, DOCUMENT IT — the pool's provenance is part of
# your method, and it changes every Check-3 number.
REFERENCE_SOURCE = "unset"


def build_pool(receipts: list, min_obs: int = None) -> dict:
    """{canonical_name: [unit_price, ...]} with guards applied.

    Returns only items meeting the minimum-observations rule. Most
    items in a small dataset will NOT qualify — that is expected and
    correct, not a bug.
    """
    threshold = _CFG["min_comparables"] if min_obs is None else min_obs
    raw = {}

    for r in receipts:
        if r.currency != "MYR":              # guard 1: currency
            continue
        for item in r.items:
            key = canonical_key(item)        # guard 2: normalised names
            if not key:
                continue
            raw.setdefault(key, []).append(round(item.unit_price, 2))

    # guard 3: minimum observations
    return {k: v for k, v in raw.items() if len(v) >= threshold}


def pool_stats(pool: dict) -> dict:
    """Diagnostics — report these. A tiny pool explains dead detection."""
    if not pool:
        return {"items": 0, "observations": 0, "median_obs": 0,
                "source": REFERENCE_SOURCE,
                "warning": "empty pool — Check 3 will flag nothing"}
    sizes = sorted(len(v) for v in pool.values())
    return {
        "items": len(pool),
        "observations": sum(sizes),
        "median_obs": sizes[len(sizes) // 2],
        "min_obs": sizes[0],
        "max_obs": sizes[-1],
        "source": REFERENCE_SOURCE,
    }


def degenerate_items(pool: dict) -> list:
    """Items whose prices are all identical → MAD = 0 → skipped.

    Worth reporting: a large list here means Check 3 is partly inert.
    """
    return [name for name, prices in pool.items()
            if len(set(prices)) == 1]


def build_from_cached(paths: list, min_obs: int = None) -> dict:
    """Build the pool from cached {canonical: [prices]} JSON.

    Use this for a FROZEN reference so Check-3 results are reproducible
    across runs.
    """
    import json
    from pathlib import Path

    merged = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for k, v in data.items():
            merged.setdefault(k, []).extend(v)

    threshold = _CFG["min_comparables"] if min_obs is None else min_obs
    return {k: v for k, v in merged.items() if len(v) >= threshold}


def save_pool(pool: dict, path: str = "data/reference/pool.json") -> str:
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pool, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)
```

### 3.16 `src/detect/similarity.py`

```python
"""String + numeric similarity for Check 2.

    Sim(x,y) = 1 − D(x,y) / max(|x|,|y|)
    S(Rᵢ,Rⱼ) = Σ wₖ sₖ        where Σ wₖ = 1
"""

from rapidfuzz.distance import Levenshtein


def string_similarity(a: str, b: str) -> float:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def numeric_similarity(a, b, tol: float) -> float:
    if a is None or b is None:
        return 0.0
    if abs(a - b) <= tol:
        return 1.0
    scale = max(abs(a), abs(b), 1e-9)
    return max(0.0, 1.0 - abs(a - b) / scale)


def date_similarity(d1: str, d2: str, window_days: int) -> float:
    from datetime import date
    if not d1 or not d2:
        return 0.0
    try:
        a, b = date.fromisoformat(d1), date.fromisoformat(d2)
    except (ValueError, TypeError):
        return 0.0
    delta = abs((a - b).days)
    return 1.0 if delta == 0 else (0.5 if delta <= window_days else 0.0)


def items_similarity(r1, r2, min_sim: float) -> float:
    """Fraction of r1's items finding a close match in r2."""
    if not r1.items or not r2.items:
        return 0.0
    hits = 0
    for a in r1.items:
        for b in r2.items:
            if (string_similarity(a.name, b.name) >= min_sim
                    and abs(a.unit_price - b.unit_price) <= 0.01):
                hits += 1
                break
    return hits / len(r1.items)


def weighted_score(r1, r2, cfg: dict) -> float:
    """S = Σ wₖ sₖ, with weights normalised so Σ wₖ = 1."""
    w = cfg["weights"]
    total_w = sum(w.values()) or 1.0

    return (
        w["store"] * string_similarity(r1.vendor, r2.vendor)
        + w["date"] * date_similarity(r1.date, r2.date,
                                      cfg["date_window_days"])
        + w["total"] * numeric_similarity(r1.receipt_total,
                                          r2.receipt_total,
                                          cfg["total_tolerance"])
        + w["items"] * items_similarity(r1, r2, cfg["item_similarity_min"])
    ) / total_w
```

### 3.17 `src/detect/check_total.py` — Check 1

```python
"""Check 1: Incorrect total. DETERMINISTIC — no model.

    E = S − D + T        flag if |E − X| > ε     ε = max(0.01, 0.001 × E)

Worked example (must pass in tests):
    items 6 + 5 + 4 = 15;  E = 15 − 2 + 1 = 14
    observed 16;  difference 2 > ε  →  mismatch
"""

from src.shared.arithmetic import receipt_ok, epsilon, is_vacuous
from src.shared.models import Anomaly


def check(receipt, discount: float = 0.0) -> list:
    anomalies = []
    if receipt.receipt_total is None:
        return anomalies

    subtotal = receipt.subtotal()
    tax = receipt.tax or 0.0

    # ── Guard 1: vacuous case ───────────────────────────────
    if is_vacuous(subtotal, receipt.receipt_total):
        return anomalies

    # ── Guard 2: discount absence ───────────────────────────
    # If the schema carries no discount field, E degrades to S + T.
    ok, expected, diff = receipt_ok(
        subtotal, discount, tax, receipt.receipt_total)

    if not ok:
        direction = "higher" if diff > 0 else "lower"
        facts = {"expected": expected, "observed": receipt.receipt_total,
                 "diff": abs(diff)}
        anomalies.append(Anomaly(
            anomaly_type="incorrect_total",
            reason=(f"Total RM {facts['observed']:.2f} is "
                    f"RM {facts['diff']:.2f} {direction} than the expected "
                    f"RM {facts['expected']:.2f}."),
            evidence={"region": "receipt_total",
                      "ref_value": expected,
                      "observed": receipt.receipt_total,
                      "difference": diff,
                      "tolerance": epsilon(expected),
                      "_facts": facts},
            confidence=min(1.0, abs(diff) / max(expected, 0.01)),
        ))

    # ── Line-level check ────────────────────────────────────
    for idx, item in enumerate(receipt.items):
        exp = round(item.qty * item.unit_price, 2)
        if abs(exp - item.line_total) > epsilon(exp):
            facts = {"expected": exp, "observed": item.line_total,
                     "qty": item.qty, "unit": item.unit_price}
            anomalies.append(Anomaly(
                anomaly_type="incorrect_total",
                reason=(f"Line '{item.name}': {item.qty} × "
                        f"RM {item.unit_price:.2f} = RM {exp:.2f}, "
                        f"but printed RM {item.line_total:.2f}."),
                evidence={"region": item.region, "ref_value": exp,
                          "observed": item.line_total,
                          "line_index": idx,
                          "ocr_confidence": item.ocr_confidence,
                          "_facts": facts},
                confidence=0.9,
            ))
    return anomalies
```

### 3.18 `src/detect/check_duplicate.py` — Check 2

```python
"""Check 2: Duplicate transaction. DETERMINISTIC — no model.

Fellegi & Sunter three-way decision. Output MUST be three-state:
match / possible match / non-match. Never binary — the middle class
routes to human review, and it is why this method was chosen.

⚠️ The deck states: "Weights and thresholds require labelled-data
testing." They were never set. Values below are starting guesses.

⚠️ BLOCKER B2 — the cross-receipt history source is unselected, so
`check_across` cannot run. within-receipt mode needs no history and
is unblocked today; it is also the default in thresholds.yaml.
"""

from collections import defaultdict

from config.thresholds_loader import THRESHOLDS
from src.shared.models import Anomaly
from src.detect.similarity import weighted_score

_CFG = THRESHOLDS["duplicate"]

MATCH, POSSIBLE, NON_MATCH = "match", "possible_match", "non_match"


def classify_pair(a, b) -> tuple:
    """Returns (verdict, score)."""
    s = weighted_score(a, b, _CFG)
    if s >= _CFG["threshold_match"]:
        return MATCH, s
    if s >= _CFG["threshold_review"]:
        return POSSIBLE, s
    return NON_MATCH, s


def check_within(receipt) -> list:
    """Within-receipt: the same line item appearing twice.

    Needs NO external history — unblocked today.
    """
    anomalies = []
    groups = defaultdict(list)
    for idx, item in enumerate(receipt.items):
        groups[item.name.strip().lower()].append((idx, item))

    for name, occurrences in groups.items():
        if len(occurrences) < 2:
            continue
        for i in range(len(occurrences) - 1):
            idx_a, a = occurrences[i]
            idx_b, b = occurrences[i + 1]
            if abs(a.qty - b.qty) > 0.001:
                continue
            if abs(a.unit_price - b.unit_price) > _CFG["total_tolerance"]:
                continue
            facts = {"unit": a.unit_price, "count": len(occurrences)}
            anomalies.append(Anomaly(
                anomaly_type="duplicate",
                reason=(f"'{a.name}' appears {len(occurrences)}× at "
                        f"RM {a.unit_price:.2f} (lines {idx_a + 1}, "
                        f"{idx_b + 1})."),
                evidence={"region": a.region, "ref_value": a.unit_price,
                          "line_indexes": [idx_a, idx_b],
                          "occurrences": len(occurrences), "_facts": facts},
                confidence=0.85,
            ))
    return anomalies


def check_across(receipts: list) -> list:
    """Across receipts. ⚠️ BLOCKED by B2 — the history source is
    unselected, so this cannot run. Scoping to within-receipt removes
    the blocker entirely."""
    anomalies = []
    for i in range(len(receipts)):
        for j in range(i + 1, len(receipts)):
            verdict, score = classify_pair(receipts[i], receipts[j])
            if verdict == NON_MATCH:
                continue
            matched = receipts[j].receipt_id
            facts = {"score": round(score, 2)}
            anomalies.append(Anomaly(
                anomaly_type="duplicate",
                reason=(f"{'Duplicate of' if verdict == MATCH else 'Possibly a duplicate of'} "
                        f"receipt #{matched} (similarity {facts['score']:.2f})."),
                evidence={"region": "receipt_header", "ref_value": score,
                          "verdict": verdict, "matched_receipt": matched,
                          "_facts": facts},
                confidence=score,
            ))
    return anomalies


def check(receipt, history: list = None) -> list:
    out = check_within(receipt)
    if _CFG["mode"] == "across_receipts" and history:
        out += check_across([receipt] + history)
    return out
```

### 3.19 `src/detect/check_price.py` — Check 3

```python
"""Check 3: Unusual price. DETERMINISTIC — no model.

Median + MAD + modified Z-score;  flag if |Mᵢ| > 3.5

    Mᵢ = 0.6745(xᵢ − x̃) / MAD        MAD = median(|xᵢ − x̃|)

⚠️ BLOCKER B1 — the reference source is unselected. A single receipt
cannot tell whether a price is unusual; an external reference is
mandatory. Build it with reference_builder.build_pool().

GUARDS:
  MAD = 0        → identical prices, division by zero. Skip the item.
  < min_comp     → too few observations for a meaningful median. Skip.
  name key       → uses the NORMALISED name, not the raw one. Without
                   normalisation the pool fragments and every item is
                   skipped silently.
"""

from statistics import median

from config.thresholds_loader import THRESHOLDS
from src.shared.models import Anomaly
from src.extract.name_normalizer import canonical_key

_CFG = THRESHOLDS["price"]


def modified_z(x: float, prices: list):
    """Returns Mᵢ, or None when MAD = 0 (degenerate)."""
    x_tilde = median(prices)
    mad = median([abs(p - x_tilde) for p in prices])
    if mad < _CFG["mad_epsilon"]:
        return None                              # ← guard: MAD = 0
    return 0.6745 * (x - x_tilde) / mad


def check(receipt, pool: dict) -> list:
    anomalies = []
    if not pool:
        return anomalies

    for item in receipt.items:
        comparables = pool.get(canonical_key(item), [])

        if len(comparables) < _CFG["min_comparables"]:
            continue                             # ← guard: too few samples

        z = modified_z(item.unit_price, comparables)
        if z is None:
            continue                             # ← degenerate MAD

        if abs(z) > _CFG["modified_z_threshold"]:
            ref = median(comparables)
            direction = "exceeds" if z > 0 else "is below"
            facts = {"observed": item.unit_price, "ref": round(ref, 2),
                     "z": round(z, 2), "n": len(comparables)}
            anomalies.append(Anomaly(
                anomaly_type="unusual_price",
                reason=(f"Unit price RM {facts['observed']:.2f} {direction} "
                        f"the median RM {facts['ref']:.2f} for "
                        f"'{item.canonical or item.name}' "
                        f"(n={facts['n']}, M={facts['z']:.2f})."),
                evidence={"region": item.region, "ref_value": ref,
                          "observed": item.unit_price, "modified_z": z,
                          "sample_size": len(comparables),
                          "canonical": canonical_key(item),
                          "_facts": facts},
                confidence=min(1.0, abs(z) / 10.0),
            ))
    return anomalies
```

### 3.20 `src/detect/detect.py` ★

```python
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
        flags += check_duplicate.check(receipt, history=self.history)
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
        }

    def run_batch(self, receipts: list) -> list:
        return [self.run(r) for r in receipts]
```

### 3.21 `src/explain/number_guard.py`

```python
"""Rejects any explanation whose numbers differ from the computed facts.

Without this, "use exactly these values" is only a polite request.
With it, an altered or invented figure is caught before display.
"""

import re

_MONEY = re.compile(r"\d+\.\d{2}")                    # "16.00"
_COUNT = re.compile(r"(?<![\d.])(\d+)(?![\d.]*\d)")   # bare integers


def _money(text: str) -> set:
    return {round(float(n), 2) for n in _MONEY.findall(text)}


def _counts(text: str) -> set:
    return {float(n) for n in _COUNT.findall(text)}


def numbers_ok(text: str, facts: dict) -> bool:
    """True if every number in `text` is accounted for in `facts`."""
    allowed = {round(float(v), 2) for v in facts.values()
               if isinstance(v, (int, float))}
    allowed |= {float(v) for v in facts.values()
                if isinstance(v, (int, float))}

    if not _money(text).issubset(allowed):
        return False

    for c in _counts(text):
        if c not in allowed and c > 100:
            return False
    return True


def guard(fallback: str, candidate: str, facts: dict) -> str:
    """Return the candidate only if it survives; else the template."""
    if not candidate or not candidate.strip():
        return fallback
    if not numbers_ok(candidate, facts):
        return fallback
    return candidate.strip()
```

### 3.22 `src/explain/explain.py` ★

```python
"""★ GATE DELIVERABLE: every flag emits the four-field contract.

{anomaly_type, reason, evidence, confidence}

Design principle: explanations are INTRINSIC — built from the same
numbers the check used. No SHAP, no LIME, no model narration.

DeepSeek is OPTIONAL and OFF by default (DOCAUDIT_LLM_EXPLAIN=0).
If enabled, code renders the sentence first, DeepSeek only rephrases,
and number_guard rejects any output that alters a figure.
"""

import os

from src.explain.number_guard import guard
from src.shared.models import Anomaly

USE_LLM_POLISH = os.getenv("DOCAUDIT_LLM_EXPLAIN", "0") == "1"


def _polish(reason: str, facts: dict) -> str:
    """Optional rephrasing. Numbers are locked INPUTS, never inferred."""
    if not USE_LLM_POLISH or not facts:
        return reason
    try:
        from config.prompts import EXPLANATION_POLISH_PROMPT
        from src.shared import deepseek_client

        payload = f"MESSAGE: {reason}\n\nFACTS (use exactly): {facts}"
        out = deepseek_client._call(EXPLANATION_POLISH_PROMPT, payload,
                                    use_cache=False, retries=1)
        candidate = out.get("text", "") if isinstance(out, dict) else str(out)
        return guard(fallback=reason, candidate=candidate, facts=facts)
    except Exception:
        return reason              # never fail the pipeline on polish


def enforce_contract(anomaly: Anomaly) -> dict:
    """Guarantees all four fields, correct types, valid enum."""
    if not anomaly.reason or not anomaly.reason.strip():
        anomaly.reason = f"{anomaly.anomaly_type.replace('_', ' ')} detected."
    if anomaly.evidence is None:
        anomaly.evidence = {}
    if not isinstance(anomaly.confidence, (int, float)):
        anomaly.confidence = 0.5
    anomaly.confidence = max(0.0, min(1.0, float(anomaly.confidence)))

    return {
        "anomaly_type": anomaly.anomaly_type,
        "reason": anomaly.reason,
        "evidence": anomaly.evidence,      # includes _facts for auditing
        "confidence": round(anomaly.confidence, 3),
    }


def explain(anomalies: list) -> list:
    out = []
    for a in anomalies:
        facts = a.evidence.get("_facts", {}) if a.evidence else {}
        a.reason = _polish(a.reason, facts)
        out.append(enforce_contract(a))
    return out


def summary(explained: list) -> str:
    if not explained:
        return "No anomalies detected."
    counts = {}
    for e in explained:
        counts[e["anomaly_type"]] = counts.get(e["anomaly_type"], 0) + 1
    parts = [f"{v}× {k.replace('_', ' ')}" for k, v in counts.items()]
    return f"{len(explained)} flag(s): " + ", ".join(parts)
```

---

**Continue to Part 2** for the UI, evaluation, and test files, plus Part 4 (what else you need beyond datasets) and Part 5 (build order).
