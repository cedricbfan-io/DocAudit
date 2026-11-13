"""Crop, deskew, denoise, orient. Runs BEFORE the reader.

PaddleOCR handles worn thermal paper well, but a skewed or badly
framed crop is still the most common cause of a bad read.
Everything here is OpenCV — no model.
"""

import cv2
import numpy as np


MAX_SIDE = 2400       # phone photos are 12–24 MP; OCR needs far less


class UnreadableImage(ValueError):
    """The upload is not an image this app can open (message is user-safe)."""


def load_image(image_path: str):
    """Any common photo format → upright BGR array.

    Pillow, not cv2.imread: OpenCV cannot open AVIF (the iPhone receipt
    'IMG_2389.AVIF' failed with 'Extraction failed: <path>'), and EXIF
    orientation is applied explicitly. HEIC works if pillow-heif is
    installed. Very large photos are scaled down to MAX_SIDE.
    """
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener   # optional
        register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            arr = np.array(im)
    except FileNotFoundError:
        raise
    except Exception as e:                      # noqa: BLE001
        raise UnreadableImage(
            f"Cannot open this file as an image ({type(e).__name__}). "
            "Use JPG, PNG, WEBP or AVIF; for iPhone HEIC photos, export "
            "as JPG first.") from e
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(image_path: str, margin: int = 8) -> str:
    """Returns a path to the cleaned image."""
    img = load_image(image_path)

    img = _deskew(img)
    img = _denoise_contrast(img)
    img = _crop_border(img, margin)

    out = image_path.rsplit(".", 1)[0] + "_clean.png"
    cv2.imwrite(out, img)
    return out


def _deskew(img):
    """Rotate by the dominant text angle — receipts are often tilted."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thr = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:      # ignore extremes
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise_contrast(img):
    """Faded thermal print benefits from contrast stretching."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _crop_border(img, margin: int):
    """Drop the dark edge from a photographed page."""
    h, w = img.shape[:2]
    if h <= 2 * margin or w <= 2 * margin:
        return img
    return img[margin:h - margin, margin:w - margin]
