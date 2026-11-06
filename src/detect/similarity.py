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
