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


_ENGINES = {}


def _get_engine(lang=None):
    """One PaddleOCR engine per language ('en', 'ch'), created lazily."""
    global _ENGINE
    lang = lang or _CFG["lang"]
    if lang == _CFG["lang"]:
        return _get_default_engine()
    if lang not in _ENGINES:
        from paddleocr import PaddleOCR
        _ENGINES[lang] = PaddleOCR(
            use_angle_cls=_CFG["use_angle_cls"], lang=lang,
            use_gpu=_CFG["use_gpu"], show_log=False,
            det_db_thresh=_CFG["det_db_thresh"],
            det_db_box_thresh=_CFG["det_db_box_thresh"])
    return _ENGINES[lang]


def _get_default_engine():
    """Lazy init — importing paddleocr is slow and memory-heavy.

    Supports PaddleOCR 2.x (the pinned version) and 3.x, whose
    constructor no longer accepts use_gpu / show_log / det_db_*.
    """
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR
        try:
            _ENGINE = PaddleOCR(
                use_angle_cls=_CFG["use_angle_cls"],
                lang=_CFG["lang"],
                use_gpu=_CFG["use_gpu"],
                show_log=False,
                det_db_thresh=_CFG["det_db_thresh"],
                det_db_box_thresh=_CFG["det_db_box_thresh"],
            )
        except (TypeError, ValueError):
            # PaddleOCR 3.x
            _ENGINE = PaddleOCR(
                lang=_CFG["lang"],
                use_textline_orientation=_CFG["use_angle_cls"],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
    return _ENGINE


def run_ocr(image_path: str, lang: str = None) -> list:
    """Image → list of regions {label, bbox_px, text, confidence}.

    One place that knows PaddleOCR's output format, for both 2.x
    ([[box, (text, conf)], ...] per page) and 3.x (result objects with
    rec_texts / rec_scores / rec_polys).
    """
    engine = _get_engine(lang)
    try:
        raw = engine.ocr(image_path, cls=True)
    except TypeError:
        raw = engine.predict(image_path)
    return _normalise_ocr(raw)


def _normalise_ocr(raw) -> list:
    regions = []
    for page in (raw or []):
        if page is None:
            continue
        res = None
        if isinstance(page, dict):
            res = page.get("res", page)
        elif hasattr(page, "json") and not isinstance(page, list):
            j = page.json
            res = j.get("res", j) if isinstance(j, dict) else None
        if res is not None and "rec_texts" in res:          # 3.x
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys", [])
            for box, text, conf in zip(polys, res["rec_texts"],
                                       res.get("rec_scores", [])):
                pts = box.tolist() if hasattr(box, "tolist") else box
                regions.append({"label": "text", "bbox_px": _to_bbox(pts),
                                "text": text, "confidence": float(conf)})
            continue
        for line in page:                                     # 2.x
            box, (text, conf) = line
            regions.append({"label": "text", "bbox_px": _to_bbox(box),
                            "text": text, "confidence": float(conf)})
    return regions


def read(image_path: str, use_cache: bool = True) -> dict:
    """Photo → {'text', 'regions', 'image_size', 'reader'}."""
    cpath = _cache_path(image_path)
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    regions = run_ocr(image_path)
    lines = [(r["bbox_px"], r["text"], r["confidence"]) for r in regions]

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


_CJK = __import__("re").compile(r"[\u4e00-\u9fff]")


def read_score(regions: list) -> float:
    """How much readable text a run produced: Σ confidence × length."""
    return sum(r["confidence"] * len(r["text"].strip()) for r in regions)


def needs_chinese(regions: list) -> bool:
    """English-model output that looks like Chinese text read badly.

    The English recogniser turns Chinese characters into short,
    low-confidence fragments. Many such regions → try the Chinese model
    (which also reads English and digits).
    """
    if not regions:
        return True
    low = [r for r in regions if r["confidence"] < 0.80]
    return len(low) / len(regions) > 0.30
