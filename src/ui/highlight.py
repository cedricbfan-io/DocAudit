"""Draws PaddleOCR boxes onto the receipt image.

This is what makes evidence visible: a flagged field points at the
exact region it came from. Because the boxes are PADDLEOCR'S OWN
detections — not model guesses — the highlight is provably correct.
"""

from PIL import Image, ImageDraw


def annotate(image_path: str, regions: list, flagged_boxes: list,
             colour=(220, 40, 40), width: int = 3):
    """Returns an annotated copy. flagged_boxes is a list of bbox strings."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    flagged = set(flagged_boxes)

    for r in regions:
        bbox = r.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        key = ",".join(str(int(v)) for v in bbox)
        if key in flagged:
            draw.rectangle(bbox, outline=colour, width=width)
    return img


def parse_box(box_str: str):
    """'10,20,110,60' → [10, 20, 110, 60]"""
    if not box_str:
        return None
    try:
        parts = [int(float(v)) for v in box_str.split(",")]
        return parts if len(parts) == 4 else None
    except (ValueError, AttributeError):
        return None
