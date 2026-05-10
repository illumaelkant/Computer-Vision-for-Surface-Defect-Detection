"""
Export YOLOv8 .pt model sang ONNX format để dùng trong production.
ONNX Runtime nhanh hơn PyTorch inference và không cần cài torch.
"""

from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(
    pt_model_path: str = "models/weights/best.pt",
    output_dir: str = "models/onnx",
    img_size: int = 640,
    opset: int = 17,           # ONNX opset version
    simplify: bool = True,     # Simplify graph (onnx-simplifier)
    dynamic: bool = False,     # Dynamic axes (batch size)
):
    """
    Export YOLOv8 sang ONNX.

    Args:
        pt_model_path: path tới file .pt
        output_dir: thư mục chứa .onnx output
        img_size: input image size
        opset: ONNX opset version (>=12 là ổn)
        simplify: dùng onnx-simplifier để tối ưu graph
        dynamic: nếu True, batch size dynamic; nếu False, fixed batch=1
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {pt_model_path}")
    model = YOLO(pt_model_path)

    logger.info(f"Exporting to ONNX (opset={opset}, simplify={simplify})...")
    onnx_path = model.export(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=False,      # FP32 (True để FP16 nếu GPU hỗ trợ)
        int8=False,      # Không quantize ở bước này
        nms=True,        # Include NMS trong ONNX graph
    )

    # Move tới output_dir
    onnx_path = Path(onnx_path)
    dest = output_dir / "defect_detector.onnx"
    onnx_path.rename(dest)
    logger.info(f"ONNX model saved: {dest}")

    # Verify ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(dest))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid")

    # Test inference với ONNX Runtime
    logger.info("Testing ONNX Runtime inference...")
    _test_onnx_inference(str(dest), img_size)

    return str(dest)


def _test_onnx_inference(onnx_path: str, img_size: int = 640):
    """Chạy thử inference để verify ONNX model hoạt động."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available_providers = ort.get_available_providers()
    use_providers = [p for p in providers if p in available_providers]

    session = ort.InferenceSession(onnx_path, providers=use_providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logger.info(f"ONNX input: name={input_name}, shape={input_shape}")

    # Tạo dummy input
    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})

    logger.info(f"ONNX inference OK. Output shapes: {[o.shape for o in outputs]}")
    logger.info(f"Providers in use: {session.get_providers()}")


if __name__ == "__main__":
    export_to_onnx()