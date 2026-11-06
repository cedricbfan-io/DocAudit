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
