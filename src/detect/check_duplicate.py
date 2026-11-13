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

import re
from collections import defaultdict

from config.thresholds_loader import THRESHOLDS
from src.shared.models import Anomaly
from src.detect.similarity import weighted_score, string_similarity
from src.extract.name_normalizer import canonical_key
from src.explain.templates import render

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


_DIGITS = re.compile(r"\d+")


def _group_key(key: str, groups) -> str:
    """Existing group whose name is a near-match for `key`, else `key`.

    Real PaddleOCR read the two identical lines of the duplicate demo
    receipt as 'Plastic' and 'Plastie'; exact keys never matched, so the
    duplicate check stayed silent. Near-match uses the same Levenshtein
    similarity and `item_similarity_min` already used across receipts.
    Numbers in the name (sizes: '1L' vs '2L') must agree exactly.
    """
    if key in groups:
        return key
    for existing in groups:
        if (_DIGITS.findall(existing) == _DIGITS.findall(key)
                and string_similarity(existing, key)
                >= _CFG["item_similarity_min"]):
            return existing
    return key


def check_within(receipt) -> list:
    """Within-receipt: the same line item appearing twice.

    Needs NO external history — unblocked today.
    """
    anomalies = []
    groups = defaultdict(list)
    for idx, item in enumerate(receipt.items):
        # group on the normalised name so OCR noise ('"Plastic' vs
        # 'Plastic', 'MILK 1L.' vs 'MILK 1L') does not hide a repeat
        key = (canonical_key(item) or item.name).strip().lower()
        groups[_group_key(key, groups)].append((idx, item))

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
            facts = {"unit": a.unit_price, "count": len(occurrences),
                     "name": a.name, "line_a": idx_a + 1, "line_b": idx_b + 1}
            anomalies.append(Anomaly(
                anomaly_type="duplicate",
                reason=render("duplicate_within", facts),
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
            facts = {"score": round(score, 2),
                     "verdict": "Duplicate of" if verdict == MATCH else "Possibly a duplicate of",
                     "matched": matched}
            anomalies.append(Anomaly(
                anomaly_type="duplicate",
                reason=render("duplicate_across", facts),
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
