"""Draws PaddleOCR boxes onto the receipt image.

This is what makes evidence visible: a flagged field points at the
exact region it came from. Because the boxes are PADDLEOCR'S OWN
detections — not model guesses — the highlight is provably correct.
"""

from PIL import Image, ImageDraw


def annotate(image_path: str, regions: list, flagged_boxes: list,
             colour=(220, 40, 40), width: int = 3):
    """Returns an annotated copy. flagged_boxes is a list of bbox strings.

    Boxes are drawn directly. (Previously a box was drawn only if it was
    byte-identical to one OCR region, so row-level evidence — an item
    spanning a number row plus its description row — never showed.)
    """
    img = Image.open(image_path).convert("RGB")
    return annotate_on(img, flagged_boxes, colour=colour, width=width)


def annotate_on(img, flagged_boxes: list, colour=(220, 40, 40), width: int = 3):
    """Draw boxes on an already-open PIL image (used for the selected row)."""
    draw = ImageDraw.Draw(img)
    seen = set()
    for b in flagged_boxes:
        box = parse_box(b)
        if not box or tuple(box) in seen:
            continue
        seen.add(tuple(box))
        x1, y1, x2, y2 = box
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2],
                       outline=colour, width=width)
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
