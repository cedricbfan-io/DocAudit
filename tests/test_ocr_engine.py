"""Locks the PaddleOCR output contract and bbox conversion."""

from src.extract.ocr_engine import (_to_bbox, box_str,
                                    low_confidence_lines, region_for_text)


def test_corner_points_to_bbox():
    """PaddleOCR gives 4 corner points → [x1, y1, x2, y2]."""
    box = [[10, 20], [110, 20], [110, 60], [10, 60]]
    assert _to_bbox(box) == [10, 20, 110, 60]


def test_bbox_handles_rotation():
    box = [[12, 18], [112, 22], [108, 62], [8, 58]]
    x1, y1, x2, y2 = _to_bbox(box)
    assert x1 == 8 and y1 == 18 and x2 == 112 and y2 == 62


def test_box_str_format():
    assert box_str([10, 20, 110, 60]) == "10,20,110,60"


def test_low_confidence_detection():
    regions = [{"confidence": 0.95, "text": "A"},
               {"confidence": 0.42, "text": "B"}]
    assert len(low_confidence_lines(regions)) == 1


def test_region_lookup_by_text():
    regions = [{"bbox_px": [0, 0, 10, 10], "text": "MILK 2L",
                "confidence": 0.9}]
    assert region_for_text(regions, "MILK") == "0,0,10,10"
    assert region_for_text(regions, "ROTI") == ""
