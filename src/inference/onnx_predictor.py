"""
Production inference sử dụng ONNX Runtime.
Không cần PyTorch, nhanh hơn và nhẹ hơn cho deployment.
"""

import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from loguru import logger


@dataclass
class Detection:
    """Một defect detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
    bbox_normalized: tuple[float, float, float, float]  # normalized [0, 1]


CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

# Màu cho từng class (BGR format cho OpenCV)
CLASS_COLORS = {
    0: (255, 100, 100),   # crazing - blue
    1: (100, 255, 100),   # inclusion - green
    2: (100, 100, 255),   # patches - red
    3: (255, 255, 100),   # pitted_surface - cyan
    4: (255, 100, 255),   # rolled-in_scale - magenta
    5: (100, 255, 255),   # scratches - yellow
}


class ONNXDefectPredictor:
    """
    Wrapper cho ONNX Runtime inference.

    Lưu ý về NMS:
    - Nếu export với nms=True: output đã include NMS, dùng _parse_output_with_nms()
    - Nếu export với nms=False: cần apply NMS thủ công, dùng _parse_output_no_nms()
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        use_gpu: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Setup ONNX Runtime session
        providers = []
        if use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]

        logger.info(f"Loading ONNX model: {model_path}")
        logger.info(f"Using providers: {providers}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        logger.info("✅ ONNX Runtime session initialized")

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        """
        Preprocess image cho YOLO input.

        Returns:
            input_tensor: (1, 3, H, W) float32 normalized
            original_shape: (H, W) của ảnh gốc
            scale: (scale_x, scale_y) để map bbox về ảnh gốc
        """
        original_h, original_w = image.shape[:2]

        # Letterbox resize (giữ aspect ratio, pad thêm)
        resized, pad_top, pad_left, scale = self._letterbox(image, self.img_size)

        # BGR → RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize [0, 255] → [0.0, 1.0]
        resized = resized.astype(np.float32) / 255.0

        # HWC → CHW → NCHW
        input_tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]

        return input_tensor, (original_h, original_w), (scale, pad_top, pad_left)

    def _letterbox(
        self,
        image: np.ndarray,
        target_size: int,
        color: tuple = (114, 114, 114),
    ) -> tuple[np.ndarray, int, int, float]:
        """Resize với letterbox (maintain aspect ratio, pad to square)."""
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Tạo canvas target_size × target_size
        canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

        # Đặt ảnh vào giữa canvas
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        return canvas, pad_top, pad_left, scale

    def postprocess(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        scale_info: tuple[float, int, int],
    ) -> list[Detection]:
        """
        Parse ONNX output → list of Detection.

        YOLOv8 với nms=True output shape: (1, N, 6) hoặc (N, 6)
        Mỗi row: [x1, y1, x2, y2, confidence, class_id]
        """
        scale, pad_top, pad_left = scale_info
        original_h, original_w = original_shape

        output = outputs[0]  # Shape: (1, N, 6) hoặc (N, 6)
        if output.ndim == 3:
            output = output[0]  # → (N, 6)

        detections = []
        for row in output:
            x1, y1, x2, y2, confidence, class_id = row

            if confidence < self.conf_threshold:
                continue

            class_id = int(class_id)
            if class_id not in CLASS_NAMES:
                continue

            # Map về pixel coords của ảnh gốc
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale

            # Clip
            x1 = max(0, min(original_w, x1))
            y1 = max(0, min(original_h, y1))
            x2 = max(0, min(original_w, x2))
            y2 = max(0, min(original_h, y2))

            # Normalized coords
            x1n = x1 / original_w
            y1n = y1 / original_h
            x2n = x2 / original_w
            y2n = y2 / original_h

            detections.append(Detection(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=float(confidence),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                bbox_normalized=(x1n, y1n, x2n, y2n),
            ))

        return detections

    def predict(self, image: np.ndarray) -> list[Detection]:
        """
        Full inference pipeline: preprocess → inference → postprocess.

        Args:
            image: BGR numpy array (từ cv2.imread hoặc bytes decoded)

        Returns:
            List of Detection objects
        """
        input_tensor, original_shape, scale_info = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, original_shape, scale_info)
        return detections

    def predict_from_file(self, image_path: str) -> tuple[np.ndarray, list[Detection]]:
        """Predict từ file path, trả về cả ảnh gốc và detections."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return image, self.predict(image)

    def draw_detections(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Vẽ bounding boxes và labels lên ảnh."""
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS.get(det.class_id, (0, 255, 0))

            # Vẽ bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{det.class_name}: {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)

            # Label text
            cv2.putText(
                result, label, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Defect count summary
        if detections:
            summary = f"Defects: {len(detections)}"
            cv2.putText(result, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return result


# Quick test
if __name__ == "__main__":
    import sys

    predictor = ONNXDefectPredictor(
        model_path="models/onnx/defect_detector.onnx",
        conf_threshold=0.5,
    )

    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/images/test/crazing_1.jpg"
    image, detections = predictor.predict_from_file(test_image_path)

    print(f"\nDetected {len(detections)} defects:")
    for det in detections:
        print(f"  [{det.class_name}] conf={det.confidence:.3f} bbox={det.bbox}")

    result = predictor.draw_detections(image, detections)
    cv2.imwrite("runs/test_prediction.jpg", result)
    print("Saved: runs/test_prediction.jpg")