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
# "RM 4.50", "RM4.50", "MYR 12.90" — price strings, not product names
_PRICE_LIKE = re.compile(r"^(RM|MYR)\s*\d", re.IGNORECASE)


def fallback_canonical(name: str) -> str:
    """Deterministic, offline, weaker than DeepSeek. Always available."""
    if not name:
        return ""
    s = name.strip().upper()
    if _ONLY_NUMBERS.match(s):
        return ""
    if _PRICE_LIKE.match(s):
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
