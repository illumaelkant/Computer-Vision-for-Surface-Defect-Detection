import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import yaml


# Mapping class name → index
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
            print(f"Unknown class: {class_name} in {xml_path}")
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

        # trong yolo label sẽ là các box, có tọa độ tâm, kích thước
        # box sẽ luôn luôn song song với trục ảnh
        annotations.append({
            "class_id": CLASS_MAP[class_name],
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })

    return annotations


def save_yolo_label(annotation: list[dict], output_path: str):
    """ghi annotation ra filt .txt yolo format"""
    with open(output_path, "w") as f:
        for ann in annotation:
            f.write(
                f"{ann['class_id']} {ann['x_center']:.6f} "
                f"{ann['y_center']:.6f} {ann['width']:.6f} "
                f"{ann['height']:.6f}\n"
            )
            
            
def prepare_dataset(
    raw_data_dir: str = "../../data/raw/NEU-DET/",
    output_dir: str = "../../data/processed",
    seed: int =  42,
):
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    
    ## find all image and its annotation
    image_path = sorted(list(raw_data_dir.rglob("*.jpg")) +
                         list(raw_data_dir.rglob("*.png")) + 
                         list(raw_data_dir.glob("*.bmp")))
    print(f"Found {len(image_path)} images")
    
    ## pair image with XML
    pairs = []
    for img_path in image_path:
        split_dir = img_path.parents[1].name
        
        xml_path = raw_data_dir / "train" / "annotations" / f"{img_path.stem}.xml"
        
        
        ## some dataset with annotation in a sperate folder
        if xml_path.exists():
            pairs.append((img_path, xml_path))
        else:
            print(f"No annotation for {img_path.name}, skipping")
            print(f"{xml_path}, {img_path}")

        
    print(f"Valid pairs: {len(pairs)}")
    
    
    ## shuffle and split
    ## tron thu tu cac pairs 1 cach ngau nhien
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n*SPLIT_RATIO["train"])
    n_val = int(n*SPLIT_RATIO["val"])
    
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    
    # copy anh va tao label yolo
    for split_name, split_pairs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok = True)
        lbl_out.mkdir(parents=True, exist_ok = True)
        
        for img_path, xml_path in tqdm(split_pairs, desc=f"Processing {split_name}"):
            shutil.copy(img_path, img_out / img_path.name)
            
            ## convert and save label
            annotations = parse_voc_xml(str(xml_path))
            label_path = lbl_out / (img_path.stem + ".txt")
            save_yolo_label(annotations, str(label_path))
            
        print(f"{split_name} : {len(split_pairs)} samples") 
        
    print("Dataset prepared successfully")
    
    
    
if __name__ == "__main__":
    prepare_dataset()           