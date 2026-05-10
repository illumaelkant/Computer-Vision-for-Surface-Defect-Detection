"""
API routes cho defect detection.

Endpoints:
    POST /api/v1/predict          - Predict từ ảnh upload
    POST /api/v1/predict/base64   - Predict từ base64 encoded image
    GET  /api/v1/classes          - Lấy danh sách classes
"""

import io
import base64
import time
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app, g
from loguru import logger
from functools import lru_cache

from api.middleware.validator import validate_image_file, validate_base64_image
from src.inference.onnx_predictor import ONNXDefectPredictor


detection_bp = Blueprint("detection", __name__)


def get_predictor() -> ONNXDefectPredictor:
    """
    Lazy-load predictor. Dùng application context để tránh load lại mỗi request.
    Singleton pattern phù hợp cho Flask với multiple workers.
    """
    if not hasattr(current_app, "_predictor"):
        current_app._predictor = ONNXDefectPredictor(
            model_path=current_app.config["MODEL_PATH"],
            conf_threshold=current_app.config["CONFIDENCE_THRESHOLD"],
            iou_threshold=current_app.config["IOU_THRESHOLD"],
        )
    return current_app._predictor


def detections_to_json(detections, include_visualization: bool = False, image: np.ndarray = None) -> dict:
    """Convert list of Detection objects sang JSON-serializable dict."""
    results = {
        "count": len(detections),
        "defects": [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 4),
                "bbox": {
                    "x1": round(det.bbox[0], 2),
                    "y1": round(det.bbox[1], 2),
                    "x2": round(det.bbox[2], 2),
                    "y2": round(det.bbox[3], 2),
                },
                "bbox_normalized": {
                    "x1": round(det.bbox_normalized[0], 6),
                    "y1": round(det.bbox_normalized[1], 6),
                    "x2": round(det.bbox_normalized[2], 6),
                    "y2": round(det.bbox_normalized[3], 6),
                },
            }
            for det in detections
        ],
        "summary": {
            cls: sum(1 for d in detections if d.class_name == cls)
            for cls in set(d.class_name for d in detections)
        },
    }

    if include_visualization and image is not None:
        predictor = get_predictor()
        vis_image = predictor.draw_detections(image, detections)
        _, buffer = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results["visualization"] = base64.b64encode(buffer).decode("utf-8")

    return results


@detection_bp.route("/predict", methods=["POST"])
def predict():
    """
    Nhận ảnh qua multipart/form-data, trả về detections.
    """
    start_time = time.time()

    # Validate
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use key 'file' in form-data."}), 400

    file = request.files["file"]
    is_valid, error_msg, image = validate_image_file(
        file,
        max_size_mb=current_app.config["MAX_IMAGE_SIZE_MB"]
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    visualize = request.form.get("visualize", "false").lower() == "true"

    # Inference
    try:
        predictor = get_predictor()
        detections = predictor.predict(image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    inference_time = (time.time() - start_time) * 1000

    response = {
        "status": "success",
        "inference_time_ms": round(inference_time, 2),
        "image_size": [image.shape[1], image.shape[0]],  # [W, H]
        **detections_to_json(detections, visualize, image),
    }

    logger.info(f"Predicted {len(detections)} defects in {inference_time:.1f}ms")
    return jsonify(response), 200


@detection_bp.route("/predict/base64", methods=["POST"])
def predict_base64():
    """
    Nhận ảnh dưới dạng base64 trong JSON body.
    """
    start_time = time.time()

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "JSON body must contain 'image' key with base64 encoded image"}), 400

    is_valid, error_msg, image = validate_base64_image(
        data["image"],
        max_size_mb=current_app.config["MAX_IMAGE_SIZE_MB"]
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    visualize = data.get("visualize", False)

    try:
        predictor = get_predictor()
        detections = predictor.predict(image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    inference_time = (time.time() - start_time) * 1000

    response = {
        "status": "success",
        "inference_time_ms": round(inference_time, 2),
        "image_size": [image.shape[1], image.shape[0]],
        **detections_to_json(detections, visualize, image),
    }

    return jsonify(response), 200


@detection_bp.route("/classes", methods=["GET"])
def get_classes():
    """Trả về danh sách defect classes."""
    from src.inference.onnx_predictor import CLASS_NAMES
    return jsonify({
        "classes": [
            {"id": k, "name": v} for k, v in sorted(CLASS_NAMES.items())
        ]
    }), 200
