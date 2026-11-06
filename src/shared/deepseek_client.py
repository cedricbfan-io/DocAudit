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

from dotenv import load_dotenv

_ENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

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
    """Call the LLM with caching, retry, and JSON enforcement.

    expect_key: if set, unwraps {key: [...]} → [...]. json_object mode
    forces an object at the top level, so arrays must be wrapped.
    """
    path = _cache_path(system_prompt, user_content)

    if use_cache and _CFG["cache"] and path.exists():
        _STATS["cache_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))

    attempts = retries if retries is not None else _CFG["max_retries"]
    last_err = None

    kwargs = dict(
        model=_CFG["model"],
        temperature=_CFG["temperature"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    if _CFG.get("json_mode", True):
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(attempts):
        try:
            resp = _get_client().chat.completions.create(**kwargs)
            parsed = json.loads(resp.choices[0].message.content.strip())
            _STATS["calls"] += 1
            break
        except Exception as e:                            # network, 429, JSON
            last_err = e
            _STATS["failures"] += 1
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)                  # 1s, 2s, 4s
    else:
        raise DeepSeekError(f"LLM call failed after {attempts} "
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
