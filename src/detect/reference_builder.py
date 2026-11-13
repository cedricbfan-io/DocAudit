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
REFERENCE_SOURCE = "curated_market_prices"


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
