
import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
from loguru import logger


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/train_config.yaml",
          dataset_config: str = "configs/dataset.yaml",
          resume: str = None):
    """
    Train YOLOv8 model.

    Args:
        config_path: path tới train_config.yaml
        dataset_config: path tới dataset.yaml
        resume: path tới checkpoint .pt để resume training (optional)
    """
    config = load_config(config_path)

    logger.info(f"Loading model: {config['model']}")

    if resume:
        # Resume từ checkpoint bị interrupt
        model = YOLO(resume)
        logger.info(f"Resuming from: {resume}")
    else:
        # Load pretrained weights (download tự động lần đầu)
        model = YOLO(config["model"])

    # Kiểm tra GPU
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU found. Training on CPU will be very slow!")

    # Bắt đầu training
    logger.info("Starting training...")
    results = model.train(
        data=dataset_config,
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=config["device"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        warmup_momentum=config["warmup_momentum"],
        warmup_bias_lr=config["warmup_bias_lr"],
        box=config["box"],
        cls=config["cls"],
        dfl=config["dfl"],
        # Augmentation
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        shear=config["shear"],
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        copy_paste=config["copy_paste"],
        # Output
        project=config["project"],
        name=config["name"],
        save=config["save"],
        save_period=config["save_period"],
        plots=config["plots"],
        resume=bool(resume),
        verbose=True,
    )

    # Best model path
    best_model_path = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_model_path}")

    # Copy best model ra models/weights/
    import shutil
    dest = Path("models/weights/best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, dest)
    logger.info(f"Copied best model to {dest}")

    return results, str(best_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    train(args.config, args.data, args.resume)