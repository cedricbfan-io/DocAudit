"""PaddleOCR PP-Structure reader. The project's only OCR path.

Returns a stable contract the rest of the system depends on:

    {"text": str, "regions": [dict], "image_size": (w, h), "reader": str}

Each region: {"label", "bbox_px", "text", "confidence"}

WHY PP-STRUCTURE:
  Receipts are a table problem more than a text problem. The line-item
  grid is the hard part, and PP-Structure's layout + table recognition
  is what makes column order recoverable at all.

WHY PIXEL BOXES MATTER:
  `bbox_px` feeds evidence highlighting in the UI. A flagged field
  points at the exact region it came from — the "supporting source
  content" the explanation contract requires.

WHY PER-BOX CONFIDENCE MATTERS:
  It is a second, independent review trigger, running BEFORE the
  arithmetic gate. Low-confidence lines route to a human even when the
  arithmetic happens to survive.

OCR IS MODEL OUTPUT → ALWAYS CACHED. Metrics must be stable.
"""

import hashlib
import json
import pathlib

from config.thresholds_loader import THRESHOLDS

_CFG = THRESHOLDS["ocr"]
_REVIEW_BELOW = _CFG["confidence"]["review_below"]
CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_ENGINE = None


def _cache_path(image_path: str) -> pathlib.Path:
    h = hashlib.sha256(
        f"{image_path}{_CFG['lang']}{_CFG['structure']}"
        f"{_CFG['det_db_thresh']}{_CFG['det_db_box_thresh']}"
        f"{_CFG['use_gpu']}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"ocr_{h}.json"


def _get_engine():
    """Lazy init — importing paddleocr is slow and memory-heavy."""
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR
        _ENGINE = PaddleOCR(
            use_angle_cls=_CFG["use_angle_cls"],
            lang=_CFG["lang"],
            use_gpu=_CFG["use_gpu"],
            show_log=False,
            det_db_thresh=_CFG["det_db_thresh"],
            det_db_box_thresh=_CFG["det_db_box_thresh"],
        )
    return _ENGINE


def read(image_path: str, use_cache: bool = True) -> dict:
    """Photo → {'text', 'regions', 'image_size', 'reader'}."""
    cpath = _cache_path(image_path)
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    raw = _get_engine().ocr(image_path, cls=True)

    regions, lines = [], []
    for page in (raw or []):
        for box, (text, conf) in (page or []):
            bbox = _to_bbox(box)
            regions.append({
                "label": "text",
                "bbox_px": bbox,
                "text": text,
                "confidence": float(conf),
            })
            lines.append((bbox, text, float(conf)))

    # Reading order: top-to-bottom, then left-to-right.
    # Receipts are single-column enough that this recovers line order.
    lines.sort(key=lambda t: (round(t[0][1] / 10), t[0][0]))
    text = "\n".join(t[1] for t in lines)

    result = {
        "text": text,
        "regions": regions,
        "image_size": _image_size(image_path),
        "reader": "paddleocr",
    }
    if use_cache:
        cpath.write_text(json.dumps(result, ensure_ascii=False),
                         encoding="utf-8")
    return result


def layout_hint(regions: list, max_regions: int = 80) -> str:
    """Compact region list for the mapper prompt.

    Region structure is what lets the mapper recover column order:
    vertical position gives rows, horizontal gives columns. Without it
    the mapper sees a flat text blob and must guess.
    """
    out = []
    for r in regions[:max_regions]:
        x1, y1, _, _ = r["bbox_px"]
        out.append(f"[y={y1} x={x1} c={r['confidence']:.2f}] {r['text']}")
    return "\n".join(out)


def low_confidence_lines(regions: list, below: float = None) -> list:
    """Regions below the threshold → human review.

    A second review trigger, independent of the arithmetic gate.
    """
    thresh = _REVIEW_BELOW if below is None else below
    return [r for r in regions if r["confidence"] < thresh]


def region_for_text(regions: list, needle: str) -> str:
    """Bbox string of the first region containing `needle`."""
    if not needle:
        return ""
    n = needle.strip().lower()
    for r in regions:
        if n and n in (r.get("text") or "").lower():
            return box_str(r["bbox_px"])
    return ""


def _to_bbox(box) -> list:
    """PaddleOCR gives 4 corner points → [x1, y1, x2, y2]."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _image_size(image_path: str) -> tuple:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size


def box_str(bbox: list) -> str:
    return ",".join(str(int(v)) for v in bbox)
