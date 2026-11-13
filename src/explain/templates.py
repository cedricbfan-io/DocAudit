"""Template-based explanation rendering.

Templates build the reason string from the same computed facts that
the check used. This makes explanations trivially auditable — the
reason IS the computation.

DeepSeek optional polish is applied in explain.py, not here. Templates
are the fallback and the source of truth.
"""

TEMPLATES = {
    "incorrect_total_receipt": (
        "Total {cur} {observed:.2f} is {cur} {diff:.2f} {direction} than "
        "the expected {cur} {expected:.2f}."
    ),
    "incorrect_total_receipt_tax": (
        "Total {cur} {observed:.2f} is {cur} {diff:.2f} {direction} than "
        "the expected {cur} {expected:.2f} (items {cur} {subtotal:.2f} "
        "+ tax {cur} {tax:.2f})."
    ),
    "incorrect_total_duplicate": (
        "Line items sum to {cur} {items_sum:.2f} but the printed total is "
        "{cur} {observed:.2f} ({cur} {diff:.2f} apart); the gap equals the "
        "repeated line {names} ({cur} {dup_amount:.2f}). Either the repeat "
        "or the TOTAL is wrong — check both."
    ),
    "incorrect_total_subtotal": (
        "Printed subtotal {cur} {observed:.2f} is {cur} {diff:.2f} {direction} "
        "than the sum of the line items, {cur} {expected:.2f}."
    ),
    "incorrect_total_line": (
        "Line '{name}': {qty} × {cur} {unit:.2f} = {cur} {expected:.2f}, "
        "but printed {cur} {observed:.2f}."
    ),
    "duplicate_within": (
        "'{name}' appears {count}× at {cur} {unit:.2f} "
        "(lines {line_a}, {line_b})."
    ),
    "duplicate_across": (
        "{verdict} receipt #{matched} (similarity {score:.2f})."
    ),
    "unusual_price": (
        "Unit price {cur} {observed:.2f} {direction} the median "
        "{cur} {ref:.2f} for '{name}' (n={n}, M={z:.2f})."
    ),
}


# Currency symbol for messages, set per receipt by Detector.run.
import contextvars as _cv
_CUR = _cv.ContextVar("docaudit_currency", default="MYR")
SYMBOL = {"MYR": "RM", "CNY": "¥", "USD": "$", "SGD": "S$", "EUR": "€",
          "GBP": "£"}


def set_currency(code: str):
    return _CUR.set(code or "MYR")


def cur_symbol() -> str:
    code = _CUR.get()
    return SYMBOL.get(code, code)


def render(template_key: str, facts: dict) -> str:
    """Render a template with the given facts. Returns the string."""
    tpl = TEMPLATES.get(template_key, "{anomaly_type} detected.")
    try:
        return tpl.format(**{"cur": cur_symbol(), **facts})
    except (KeyError, ValueError):
        return tpl
