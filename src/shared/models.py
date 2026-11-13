"""Dataclasses mirroring the two frozen JSON contracts."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Region:
    """One OCR region. PaddleOCR gives pixel boxes natively."""
    label: str                      # text | title | table | figure
    bbox_px: list                   # [x1, y1, x2, y2] in PIXELS
    text: str = ""
    confidence: float = 1.0


@dataclass
class LineItem:
    name: str                       # raw name as read from the receipt
    qty: float
    unit_price: float
    line_total: float
    mismatch: bool = False          # set by the mapper, confirmed by the gate

    canonical: str = ""             # normalised name — used by Check 3
    raw_text: str = ""              # original OCR row text
    region: str = ""                # bbox string, for evidence highlighting
    ocr_confidence: float = 1.0
    review: str = "unreviewed"      # unreviewed | review_required | corrected | accepted | rejected
    reason: str = ""                # why review was requested


@dataclass
class Receipt:
    vendor: Optional[str]
    date: Optional[str]
    currency: str = "MYR"
    items: list = field(default_factory=list)
    receipt_total: Optional[float] = None
    tax: Optional[float] = None
    printed_subtotal: Optional[float] = None    # SUBTOTAL / TOTAL QTY row, as printed

    receipt_id: str = ""
    image_path: str = ""
    image_size: Optional[tuple] = None
    regions: list = field(default_factory=list)     # list[Region]
    duplicate_flag: bool = False
    provenance: dict = field(default_factory=dict)

    def subtotal(self) -> float:
        return round(sum(i.line_total for i in self.items), 2)

    def needs_review(self) -> bool:
        return any(i.review == "review_required" for i in self.items) \
            or bool(self.provenance.get("_total_gap"))


@dataclass
class Anomaly:
    """The four-field contract. Every flag emits exactly this."""
    anomaly_type: str               # incorrect_total | duplicate | unusual_price
    reason: str
    evidence: dict
    confidence: float

    ALLOWED = ("incorrect_total", "duplicate", "unusual_price")

    def __post_init__(self):
        if self.anomaly_type not in self.ALLOWED:
            raise ValueError(f"bad anomaly_type: {self.anomaly_type}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
