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
from src.explain.templates import render

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
                     "z": round(z, 2), "n": len(comparables),
                     "direction": direction,
                     "name": item.canonical or item.name}
            anomalies.append(Anomaly(
                anomaly_type="unusual_price",
                reason=render("unusual_price", facts),
                evidence={"region": item.region, "ref_value": ref,
                          "observed": item.unit_price, "modified_z": z,
                          "sample_size": len(comparables),
                          "canonical": canonical_key(item),
                          "_facts": facts},
                confidence=min(1.0, abs(z) / 10.0),
            ))
    return anomalies
