"""Single import point for every tunable number.

Nothing reads the YAML directly except this module. When you tune a
threshold, you change one file and every check follows.
"""

import pathlib
import yaml

_PATH = pathlib.Path(__file__).parent / "thresholds.yaml"

with _PATH.open(encoding="utf-8") as f:
    THRESHOLDS = yaml.safe_load(f)


def get(path: str, default=None):
    """Dot-path access: get('duplicate.weights.store')."""
    node = THRESHOLDS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
