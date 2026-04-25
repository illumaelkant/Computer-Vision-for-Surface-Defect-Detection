# 🏭 Hướng Dẫn Triển Khai: Computer Vision for Surface Defect Detection

> **Mục tiêu:** Xây dựng hệ thống phát hiện lỗi bề mặt công nghiệp (scratches, dents, cracks...) đạt 98.5% mAP, export sang ONNX và deploy qua Flask REST API.
>
> **Đối tượng:** Người đã biết Python cơ bản, muốn triển khai một project CV end-to-end hoàn chỉnh.
>
> **Stack:** Python 3.10 · YOLOv8 (Ultralytics) · OpenCV · ONNX Runtime · Flask · Docker

---

## MỤC LỤC

1. [Tổng quan kiến trúc hệ thống](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Thiết lập môi trường](#3-thiết-lập-môi-trường)
4. [Thu thập dữ liệu](#4-thu-thập-dữ-liệu)
5. [Tiền xử lý và Data Augmentation](#5-tiền-xử-lý-và-data-augmentation)
6. [Cấu hình và Training YOLOv8](#6-cấu-hình-và-training-yolov8)
7. [Đánh giá mô hình](#7-đánh-giá-mô-hình)
8. [Export sang ONNX](#8-export-sang-onnx)
9. [Inference pipeline](#9-inference-pipeline)
10. [Flask REST API](#10-flask-rest-api)
11. [Docker & Deployment](#11-docker--deployment)
12. [Logging & Monitoring](#12-logging--monitoring)
13. [Những lỗi phổ biến và cách xử lý](#13-những-lỗi-phổ-biến)

---

## 1. Tổng Quan Kiến Trúc

```
[Camera / Image Input]
        │
        ▼
[Preprocessing Module]   ← resize, normalize, augment (training only)
        │
        ▼
[YOLOv8 Model]           ← fine-tuned trên industrial defect dataset
        │
        ▼
[Post-processing]        ← NMS, confidence filter, class mapping
        │
        ▼
[ONNX Runtime Engine]    ← production inference (không cần PyTorch)
        │
        ▼
[Flask REST API]         ← nhận ảnh → trả về JSON với bounding boxes
        │
        ▼
[Client / Dashboard]     ← hiển thị kết quả QC
```

**Luồng training:**
```
Raw Data → EDA → Augmentation → YOLO Training → Evaluation → Export ONNX
```

**Luồng inference (production):**
```
HTTP POST (image) → Flask → ONNX Runtime → JSON Response
```

---

## 2. Cấu Trúc Thư Mục

Tạo đúng cấu trúc này trước, không đặt file linh tinh:

```
surface_defect_detection/
│
├── data/
│   ├── raw/                        # Dữ liệu gốc download về, KHÔNG sửa
│   │   ├── NEU-DET/
│   │   │   ├── images/
│   │   │   └── annotations/       # XML format (VOC) hoặc YOLO .txt
│   │   └── custom/                 # Ảnh tự chụp nếu có
│   │
│   ├── processed/                  # Sau khi chạy preprocess script
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/                 # YOLO format (.txt)
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── augmented/                  # Output của augmentation (optional)
│
├── configs/
│   ├── dataset.yaml                # YOLO dataset config
│   └── train_config.yaml           # Hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_dataset.py      # Convert VOC→YOLO, split train/val/test
│   │   ├── augmentation.py         # Albumentations pipeline
│   │   └── dataloader.py           # Custom dataloader nếu cần
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # Script chính để train
│   │   └── callbacks.py            # Custom YOLO callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py             # mAP, confusion matrix, visualize
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py            # Inference với PyTorch model
│   │   └── onnx_predictor.py       # Inference với ONNX Runtime
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualize.py            # Vẽ bounding box lên ảnh
│       └── logger.py               # Logging config
│
├── api/
│   ├── __init__.py
│   ├── app.py                      # Flask app chính
│   ├── routes/
│   │   ├── __init__.py
│   │   └── detection.py            # Route /predict
│   └── middleware/
│       ├── __init__.py
│       └── validator.py            # Validate input image
│
├── models/
│   ├── weights/                    # .pt files sau training
│   └── onnx/                       # .onnx files sau export
│
├── runs/                           # YOLO tự tạo, chứa experiment logs
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_augmentation_test.ipynb
│   └── 03_result_analysis.ipynb
│
├── tests/
│   ├── test_api.py
│   └── test_inference.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

**Tạo cấu trúc này tự động:**

```bash
#!/bin/bash
# run_setup.sh - chạy 1 lần để tạo cấu trúc

mkdir -p data/{raw/{NEU-DET/{images,annotations},custom},processed/{images/{train,val,test},labels/{train,val,test}},augmented}
mkdir -p configs
mkdir -p src/{data,training,evaluation,inference,utils}
mkdir -p api/{routes,middleware}
mkdir -p models/{weights,onnx}
mkdir -p notebooks tests runs

# Tạo __init__.py
touch src/__init__.py src/data/__init__.py src/training/__init__.py
touch src/evaluation/__init__.py src/inference/__init__.py src/utils/__init__.py
touch api/__init__.py api/routes/__init__.py api/middleware/__init__.py

echo "✅ Structure created"
```

---

## 3. Thiết Lập Môi Trường

### 3.1 Tạo virtual environment

```bash
# Dùng conda (khuyến nghị cho ML projects)
conda create -n defect_det python=3.10 -y
conda activate defect_det

# Hoặc dùng venv
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3.2 `requirements.txt`

```txt
# Core ML
ultralytics==8.2.0
torch==2.2.0
torchvision==0.17.0
onnxruntime==1.17.1
onnx==1.15.0

# Computer Vision & Image Processing
opencv-python==4.9.0.80
Pillow==10.2.0
albumentations==1.4.0

# Data Science
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.0

# API
Flask==3.0.2
Flask-CORS==4.0.0
Werkzeug==3.0.1
gunicorn==21.2.0

# Utilities
python-dotenv==1.0.1
PyYAML==6.0.1
tqdm==4.66.2
loguru==0.7.2

# Dev/Test
pytest==8.0.2
httpx==0.27.0
```

```bash
pip install -r requirements.txt
```

### 3.3 `.env.example`

```env
# Copy thành .env và điền giá trị thực
MODEL_PATH=models/onnx/defect_detector.onnx
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
MAX_IMAGE_SIZE_MB=10
FLASK_ENV=development
FLASK_PORT=5000
LOG_LEVEL=INFO
```

---

## 4. Thu Thập Dữ Liệu

### 4.1 Dataset công khai được khuyến nghị

**Option 1 (Khuyến nghị nhất): NEU Surface Defect Database**
- Link: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
- 6 classes: Rolled-in Scale (Rs), Patches (Pa), Crazing (Cr), Pitted Surface (Ps), Inclusion (In), Scratches (Sc)
- 1.800 ảnh grayscale 200×200px
- Format: VOC XML annotations

**Option 2: MVTec AD (Anomaly Detection)**
- Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
- 15 loại object/texture, 5.354 ảnh
- Phù hợp nếu muốn thêm bài toán anomaly detection

**Option 3: Kaggle - Steel Defect Detection**
- Link: https://www.kaggle.com/c/severstal-steel-defect-detection
- Ảnh thép cuộn lớn (256×1600px), 4 loại defect
- Format: RLE mask (cần convert sang bounding box)

**Option 4: DAGM 2007**
- Link: https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html
- 10 classes texture defects
- Grayscale, có weak supervision labels

> **Với project này, dùng NEU-DET là đủ và phù hợp nhất.**

### 4.2 Download NEU-DET

```bash
# Tải thủ công từ website, sau đó giải nén vào:
# data/raw/NEU-DET/

# Hoặc dùng script (nếu có Kaggle API key):
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
unzip neu-surface-defect-database.zip -d data/raw/NEU-DET/
```

### 4.3 Kiểm tra dữ liệu sau download

```python
# notebooks/01_eda.ipynb - chạy cell này để verify

import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

data_dir = Path("data/raw/NEU-DET")

# Đếm ảnh theo class
class_counts = {}
for split in ["train", "validation"]:  # tuỳ dataset structure
    img_dir = data_dir / split / "images"
    if img_dir.exists():
        for img_path in img_dir.glob("*.jpg"):
            class_name = img_path.stem.split("_")[0]  # NEU format: Rs_001.jpg
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

print("Class distribution:", class_counts)

# Visualize sample images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
class_dirs = list(data_dir.glob("**/images"))
# ... (vẽ sample từ mỗi class)
```

---

## 5. Tiền Xử Lý và Data Augmentation

### 5.1 Convert VOC XML → YOLO format

**File: `src/data/prepare_dataset.py`**

```python
"""
Convert Pascal VOC XML annotations sang YOLO format (.txt)
và split thành train/val/test.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import yaml


# Mapping class name → index (phải nhất quán với dataset.yaml)
CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}

SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}


def parse_voc_xml(xml_path: str) -> list[dict]:
    """Parse một file XML VOC, trả về list của các object annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    annotations = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower().strip()
        if class_name not in CLASS_MAP:
            print(f"⚠️  Unknown class: {class_name} in {xml_path}")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert sang YOLO format (normalized center x, center y, w, h)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clip để tránh giá trị out-of-bound do annotation lỗi
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.001, min(1.0, width))
        height = max(0.001, min(1.0, height))

        annotations.append({
            "class_id": CLASS_MAP[class_name],
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })

    return annotations


def save_yolo_label(annotations: list[dict], output_path: str):
    """Ghi annotations ra file .txt YOLO format."""
    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(
                f"{ann['class_id']} {ann['x_center']:.6f} "
                f"{ann['y_center']:.6f} {ann['width']:.6f} "
                f"{ann['height']:.6f}\n"
            )


def prepare_dataset(
    raw_data_dir: str = "data/raw/NEU-DET",
    output_dir: str = "data/processed",
    seed: int = 42,
):
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    # Tìm tất cả ảnh và annotation tương ứng
    image_paths = sorted(list(raw_data_dir.rglob("*.jpg")) +
                         list(raw_data_dir.rglob("*.png")) +
                         list(raw_data_dir.rglob("*.bmp")))

    print(f"Found {len(image_paths)} images")

    # Pair ảnh với XML
    pairs = []
    for img_path in image_paths:
        # Tìm file XML tương ứng (cùng tên, khác extension)
        xml_path = img_path.with_suffix(".xml")
        # Một số dataset để annotations ở thư mục riêng
        if not xml_path.exists():
            xml_path = Path(str(img_path).replace("images", "annotations")).with_suffix(".xml")
        if xml_path.exists():
            pairs.append((img_path, xml_path))
        else:
            print(f"⚠️  No annotation for {img_path.name}, skipping")

    print(f"Valid pairs: {len(pairs)}")

    # Shuffle và split
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # Copy ảnh và tạo label YOLO
    for split_name, split_pairs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in tqdm(split_pairs, desc=f"Processing {split_name}"):
            # Copy ảnh
            shutil.copy(img_path, img_out / img_path.name)

            # Convert và save label
            annotations = parse_voc_xml(str(xml_path))
            label_path = lbl_out / (img_path.stem + ".txt")
            save_yolo_label(annotations, str(label_path))

        print(f"  {split_name}: {len(split_pairs)} samples")

    print("\n✅ Dataset prepared successfully")


if __name__ == "__main__":
    prepare_dataset()
```

### 5.2 Data Augmentation pipeline

**File: `src/data/augmentation.py`**

```python
"""
Augmentation pipeline dùng Albumentations.
Chú ý: YOLOv8 đã có built-in augmentation (mosaic, mixup, hsv, flip...),
script này dùng để pre-augment trên dataset nhỏ hoặc làm offline augmentation.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """
    Pipeline augmentation cho training.
    bbox_params quan trọng: YOLO format cần format='yolo'.
    """
    return A.Compose(
        [
            # --- Geometric transforms ---
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.2),

            # --- Color/Brightness transforms ---
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3,
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),          # Cải thiện contrast cục bộ
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # --- Noise transforms ---
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),

            # --- Blur transforms (simulate camera defocus) ---
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # --- Occlusion simulation ---
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=2,
                fill_value=0,
                p=0.2,
            ),

            # --- Final resize ---
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,  # YOLO default padding value
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,      # Drop box nếu bị che >70%
            min_area=100,            # Drop box nếu area quá nhỏ sau augment
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Val/Test: chỉ resize, không augment."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def augment_dataset_offline(
    processed_dir: str = "data/processed",
    output_dir: str = "data/augmented",
    augment_factor: int = 3,       # Mỗi ảnh train tạo thêm N ảnh augmented
    img_size: int = 640,
):
    """
    Offline augmentation: tạo N bản augmented cho mỗi ảnh training.
    Dùng khi dataset quá nhỏ (<500 ảnh).
    Nếu dataset đủ lớn, bỏ qua bước này — YOLO online augmentation là đủ.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    transform = get_train_transforms(img_size)

    for split in ["train"]:  # Chỉ augment train set
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        for img_path in tqdm(img_paths, desc=f"Augmenting {split}"):
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            bboxes = []
            class_labels = []

            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_labels.append(int(parts[0]))
                            bboxes.append([float(x) for x in parts[1:]])

            # Copy ảnh gốc
            cv2.imwrite(
                str(out_img_dir / img_path.name),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            if lbl_path.exists():
                import shutil
                shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)

            # Tạo N bản augmented
            for i in range(augment_factor):
                try:
                    result = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )
                    aug_image = result["image"]
                    aug_bboxes = result["bboxes"]
                    aug_labels = result["class_labels"]

                    # Bỏ qua nếu không còn bbox nào
                    if not aug_bboxes and bboxes:
                        continue

                    aug_name = f"{img_path.stem}_aug{i}"
                    cv2.imwrite(
                        str(out_img_dir / f"{aug_name}.jpg"),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                    )

                    with open(out_lbl_dir / f"{aug_name}.txt", "w") as f:
                        for cls, bbox in zip(aug_labels, aug_bboxes):
                            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

                except Exception as e:
                    print(f"Augmentation failed for {img_path.name}: {e}")

    print("✅ Offline augmentation complete")


if __name__ == "__main__":
    augment_dataset_offline()
```

---

## 6. Cấu Hình và Training YOLOv8

### 6.1 Dataset config

**File: `configs/dataset.yaml`**

```yaml
# Đường dẫn tuyệt đối hoặc tương đối từ thư mục root của project
path: data/processed          # root dir của dataset
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 6

# Class names - phải đúng thứ tự với CLASS_MAP trong prepare_dataset.py
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

### 6.2 Training hyperparameters

**File: `configs/train_config.yaml`**

```yaml
# Training configuration
model: yolov8s.pt          # Start từ pretrained: n/s/m/l/x (n=nano, s=small, ...)
                            # s là điểm cân bằng tốt giữa speed và accuracy
epochs: 100
patience: 20               # Early stopping: dừng nếu không cải thiện sau 20 epoch
batch: 16                  # Giảm xuống 8 nếu OOM
imgsz: 640
device: 0                  # 0 = GPU 0, 'cpu' = CPU, '0,1' = multi-GPU
workers: 4

# Optimizer
optimizer: AdamW
lr0: 0.001                 # Initial learning rate
lrf: 0.01                  # Final LR = lr0 * lrf
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights
box: 7.5
cls: 0.5
dfl: 1.5

# Augmentation (YOLO built-in - đây là production settings)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
shear: 2.0
perspective: 0.0
flipud: 0.3
fliplr: 0.5
mosaic: 1.0               # Mosaic augmentation - rất hiệu quả
mixup: 0.1
copy_paste: 0.1

# Output
project: runs/train
name: defect_detector_v1
save: true
save_period: 10           # Save checkpoint mỗi 10 epoch
plots: true               # Generate training plots
```

### 6.3 Training script

**File: `src/training/train.py`**

```python
"""
Main training script.
Usage: python src/training/train.py
"""

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
```

**Chạy training:**

```bash
# Lần đầu train
python src/training/train.py

# Resume nếu bị interrupt
python src/training/train.py --resume runs/train/defect_detector_v1/weights/last.pt
```

---

## 7. Đánh Giá Mô Hình

**File: `src/evaluation/evaluate.py`**

```python
"""
Evaluate trained model trên test set.
Tính mAP@0.5, mAP@0.5:0.95, precision, recall, và vẽ confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import yaml


CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def evaluate_model(
    model_path: str = "models/weights/best.pt",
    dataset_config: str = "configs/dataset.yaml",
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
```

```bash
# Chạy evaluation
python src/evaluation/evaluate.py
```

---

## 8. Export Sang ONNX

**File: `src/training/export_onnx.py`**

```python
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
    logger.info("✅ ONNX model is valid")

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

    logger.info(f"✅ ONNX inference OK. Output shapes: {[o.shape for o in outputs]}")
    logger.info(f"Providers in use: {session.get_providers()}")


if __name__ == "__main__":
    export_to_onnx()
```

```bash
# Export
python src/training/export_onnx.py
```

---

## 9. Inference Pipeline

**File: `src/inference/onnx_predictor.py`**

```python
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
```

---

## 10. Flask REST API

### 10.1 App chính

**File: `api/app.py`**

```python
"""
Flask REST API cho Surface Defect Detection.
"""

import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from loguru import logger

from api.routes.detection import detection_bp


def create_app() -> Flask:
    """Application factory pattern."""
    load_dotenv()

    app = Flask(__name__)
    CORS(app)

    # Config từ env
    app.config["MODEL_PATH"] = os.getenv("MODEL_PATH", "models/onnx/defect_detector.onnx")
    app.config["CONFIDENCE_THRESHOLD"] = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    app.config["IOU_THRESHOLD"] = float(os.getenv("IOU_THRESHOLD", "0.45"))
    app.config["MAX_IMAGE_SIZE_MB"] = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_IMAGE_SIZE_MB"] * 1024 * 1024)

    # Register blueprints
    app.register_blueprint(detection_bp, url_prefix="/api/v1")

    # Health check
    @app.route("/health")
    def health():
        return {"status": "healthy", "version": "1.0.0"}, 200

    logger.info(f"App initialized. Model: {app.config['MODEL_PATH']}")
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
```

### 10.2 Detection route

**File: `api/routes/detection.py`**

```python
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

    Form params:
        file: image file (jpg/png/bmp)
        visualize: 'true'/'false' - có trả về ảnh đã annotated không (default: false)

    Response:
        {
            "status": "success",
            "inference_time_ms": 42.5,
            "image_size": [640, 480],
            "count": 3,
            "defects": [...],
            "summary": {...}
        }
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

    Request body:
        {
            "image": "<base64_encoded_image>",
            "visualize": false
        }
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
```

### 10.3 Middleware validator

**File: `api/middleware/validator.py`**

```python
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
```

### 10.4 Chạy API

```bash
# Development
python api/app.py

# Production (gunicorn - 4 workers)
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 60 \
  --access-logfile - \
  "api.app:create_app()"
```

### 10.5 Test API với curl

```bash
# Health check
curl http://localhost:5000/health

# Predict từ file
curl -X POST \
  -F "file=@data/processed/images/test/scratches_001.jpg" \
  -F "visualize=false" \
  http://localhost:5000/api/v1/predict

# Predict với base64
IMAGE_B64=$(base64 -w 0 data/processed/images/test/scratches_001.jpg)
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"visualize\": false}" \
  http://localhost:5000/api/v1/predict/base64
```

**Response mẫu:**

```json
{
    "status": "success",
    "inference_time_ms": 38.4,
    "image_size": [200, 200],
    "count": 2,
    "defects": [
        {
            "class_id": 5,
            "class_name": "scratches",
            "confidence": 0.9231,
            "bbox": {"x1": 42.1, "y1": 67.3, "x2": 158.9, "y2": 134.2},
            "bbox_normalized": {"x1": 0.21, "y1": 0.34, "x2": 0.79, "y2": 0.67}
        }
    ],
    "summary": {"scratches": 2}
}
```

---

## 11. Docker & Deployment

### 11.1 Dockerfile

```dockerfile
# --- Build stage ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# --- Runtime stage ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime dependencies cho OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages từ builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/onnx/ ./models/onnx/

# .env sẽ được mount vào lúc runtime, không COPY vào image
COPY .env.example .env

# Non-root user (security best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", \
     "--workers", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "api.app:create_app()"]
```

### 11.2 docker-compose.yml

```yaml
version: "3.9"

services:
  defect-api:
    build:
      context: .
      target: runtime
    container_name: defect_detection_api
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      # Mount model file (không bake vào image để dễ update model)
      - ./models/onnx/defect_detector.onnx:/app/models/onnx/defect_detector.onnx:ro
      # Log output
      - ./logs:/app/logs
    restart: unless-stopped
    # Resource limits (điều chỉnh theo server)
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
        reservations:
          memory: 512M

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - defect-api
    restart: unless-stopped
```

### 11.3 Build và chạy

```bash
# Build image
docker build -t defect-detection:latest .

# Chạy với docker-compose
docker-compose up -d

# Xem logs
docker-compose logs -f defect-api

# Stop
docker-compose down
```

---

## 12. Logging & Monitoring

**File: `src/utils/logger.py`**

```python
"""Cấu hình Loguru cho toàn bộ project."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """
    Setup Loguru logger.
    Gọi hàm này 1 lần trong app entry point.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Console handler (màu mè)
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler (structured cho production)
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,  # Thread-safe
    )

    return logger
```

---

## 13. Những Lỗi Phổ Biến

### ❌ Lỗi 1: CUDA out of memory trong training

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Fix:**
```yaml
# Trong train_config.yaml
batch: 8          # Giảm từ 16 xuống 8
imgsz: 416        # Giảm từ 640 xuống 416
```

---

### ❌ Lỗi 2: Bounding box bị lệch sau inference

**Nguyên nhân:** Quên account cho letterbox padding khi map bbox về ảnh gốc.

**Fix:** Đảm bảo dùng `scale`, `pad_top`, `pad_left` từ `_letterbox()` để reverse transform (đã handle trong `onnx_predictor.py` ở trên).

---

### ❌ Lỗi 3: mAP thấp dù loss nhỏ

**Checklist:**
1. Kiểm tra CLASS_MAP trong `prepare_dataset.py` có khớp với `dataset.yaml` không
2. Kiểm tra label format: dòng trong `.txt` phải là `class_id x_center y_center width height`, tất cả normalized [0, 1]
3. Dùng YOLO built-in check: `python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').val(data='configs/dataset.yaml', split='val')"`

---

### ❌ Lỗi 4: ONNX output shape không đúng

**Nguyên nhân:** Xuất với `nms=True` nhưng parse output sai.

**Debug:**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/onnx/defect_detector.onnx")
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
for out in session.get_outputs():
    print(f"Output: {out.name}, shape={out.shape}")
```

---

### ❌ Lỗi 5: Flask OOM khi nhiều concurrent requests

**Nguyên nhân:** Mỗi request đang load model lại (không dùng singleton).

**Fix:** Đảm bảo `get_predictor()` dùng `current_app._predictor` như đã code ở trên, hoặc init predictor 1 lần trong `create_app()` và attach vào `app` object.

---

### ❌ Lỗi 6: Augmentation làm mất bounding box

**Nguyên nhân:** Sau rotate/crop, bbox ra ngoài ảnh.

**Fix:** Albumentations đã xử lý nếu set đúng `bbox_params`:
```python
bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3,   # Drop box nếu bị che > 70%
)
```

---

## Checklist Hoàn Thành Project

```
✅ Cấu trúc thư mục tạo xong
✅ requirements.txt installed
✅ Dataset download (NEU-DET) vào data/raw/
✅ prepare_dataset.py chạy thành công
✅ dataset.yaml cấu hình đúng class names và paths
✅ Training chạy được (ít nhất 50 epoch)
✅ mAP@0.5 ≥ 95% trên val set
✅ Export ONNX thành công và verify được
✅ onnx_predictor.py test OK trên 1 ảnh test
✅ Flask API /predict trả về đúng format JSON
✅ Docker build và run thành công
✅ curl test API trả về kết quả đúng
```

---

*Tài liệu này là hướng dẫn triển khai đủ để build từ zero đến production.*
*Nếu bạn tự thu thập dữ liệu thực tế (ảnh từ camera công nghiệp), phần preprocess cần thêm bước chuẩn hóa nhiễu cảm biến và calibration — đó là một topic khác.*
