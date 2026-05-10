

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import yaml


CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def evaluate_model(
    model_path: str = "../../models/weights/best.pt",
    dataset_config: str = "../../configs/dataset.yaml",
    split: str = "test",
    output_dir: str = "runs/evaluation",
    conf: float = 0.25,
    iou: float = 0.45,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    logger.info(f"Evaluating on {split} split...")
    metrics = model.val(
        data=dataset_config,
        split=split,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        project=str(output_dir),
        name="results",
        verbose=True,
    )

    # In kết quả
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:        {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"mAP@0.5:0.95:   {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print(f"Precision:      {metrics.box.mp:.4f}")
    print(f"Recall:         {metrics.box.mr:.4f}")
    print("=" * 60)

    # Per-class metrics
    print("\nPer-class AP@0.5:")
    for i, (cls_name, ap) in enumerate(zip(CLASS_NAMES, metrics.box.ap50)):
        print(f"  {cls_name:<20}: {ap:.4f} ({ap * 100:.2f}%)")

    return metrics


def visualize_predictions(
    model_path: str = "models/weights/best.pt",
    test_images_dir: str = "data/processed/images/test",
    output_dir: str = "runs/predictions",
    conf: float = 0.35,
    n_samples: int = 20,
):
    """Visualize predictions trên test images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    test_images = list(Path(test_images_dir).glob("*.jpg"))[:n_samples]

    results = model.predict(
        source=test_images,
        conf=conf,
        save=True,
        project=str(output_dir),
        name="vis",
        line_width=2,
    )

    logger.info(f"Saved predictions to {output_dir}/vis/")
    return results


if __name__ == "__main__":
    evaluate_model()
    visualize_predictions()