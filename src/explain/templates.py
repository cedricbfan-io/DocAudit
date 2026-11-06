"""Template-based explanation rendering.

Templates build the reason string from the same computed facts that
the check used. This makes explanations trivially auditable — the
reason IS the computation.

DeepSeek optional polish is applied in explain.py, not here. Templates
are the fallback and the source of truth.
"""

TEMPLATES = {
    "incorrect_total_receipt": (
        "Total RM {observed:.2f} is RM {diff:.2f} {direction} than "
        "the expected RM {expected:.2f}."
    ),
    "incorrect_total_line": (
        "Line '{name}': {qty} × RM {unit:.2f} = RM {expected:.2f}, "
        "but printed RM {observed:.2f}."
    ),
    "duplicate_within": (
        "'{name}' appears {count}× at RM {unit:.2f} "
        "(lines {line_a}, {line_b})."
    ),
    "duplicate_across": (
        "{verdict} receipt #{matched} (similarity {score:.2f})."
    ),
    "unusual_price": (
        "Unit price RM {observed:.2f} {direction} the median "
        "RM {ref:.2f} for '{name}' (n={n}, M={z:.2f})."
    ),
}


def render(template_key: str, facts: dict) -> str:
    """Render a template with the given facts. Returns the string."""
    tpl = TEMPLATES.get(template_key, "{anomaly_type} detected.")
    try:
        return tpl.format(**facts)
    except (KeyError, ValueError):
        return tpl
