"""Input validation cho API."""

import io
import base64
import cv2
import numpy as np
from PIL import Image
from loguru import logger


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}


def _decode_image(image_bytes: bytes) -> tuple[bool, str, np.ndarray | None]:
    """Decode raw bytes sang OpenCV BGR numpy array."""
    try:
        # Dùng PIL để handle nhiều format hơn
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA → RGB (YOLO không cần alpha channel)
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")

        # PIL (RGB) → OpenCV (BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return True, "", image

    except Exception as e:
        return False, f"Cannot decode image: {str(e)}", None


def validate_image_file(
    file,
    max_size_mb: float = 10.0,
) -> tuple[bool, str, np.ndarray | None]:
    """Validate file upload từ Flask request."""
    if file.filename == "":
        return False, "Empty filename", None

    # Check extension
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}", None

    # Read bytes
    image_bytes = file.read()

    # Check size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB > {max_size_mb}MB", None

    return _decode_image(image_bytes)


def validate_base64_image(
    b64_string: str,
    max_size_mb: float = 10.0,
) -> tuple[bool, str, np.ndarray | None]:
    """Validate base64-encoded image string."""
    # Xoá data URL prefix nếu có (data:image/jpeg;base64,...)
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_string)
    except Exception:
        return False, "Invalid base64 string", None

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Image too large: {size_mb:.1f}MB > {max_size_mb}MB", None

    return _decode_image(image_bytes)