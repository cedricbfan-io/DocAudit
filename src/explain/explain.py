"""★ GATE DELIVERABLE: every flag emits the four-field contract.

{anomaly_type, reason, evidence, confidence}

Design principle: explanations are INTRINSIC — built from the same
numbers the check used. No SHAP, no LIME, no model narration.

DeepSeek is OPTIONAL and OFF by default (DOCAUDIT_LLM_EXPLAIN=0).
If enabled, code renders the sentence first, DeepSeek only rephrases,
and number_guard rejects any output that alters a figure.
"""

import os
import pathlib

from dotenv import load_dotenv

_ENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

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
