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
            # Mo phong anh bi che boi cac vat the
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
        
        ## config để label biến đổi tương ứng
        bbox_params = A.BboxParams(
            format="yolo",
            label_fields = ["class_labels"],
            min_visibility = 0.3,
            min_area = 100,
        ),
    )
    
    
def get_val_transforms(img_size: int = 640) -> A.Compose:
    """ Val, test: only resize, no argument"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height = img_size,
                min_width = img_size,
                border_mode = cv2.BORDER_CONSTANT,
                value = 114,
            ),
        ],
            
        bbox_params = A.BboxParams(
            format = "yolo",
            label_fields = ["class_labels"],
        ),
    )
    
def augment_dataset_offline(
    processed_dir: str = "../../data/processed",
    output_dir: str = "../../data/augmented",
    ## moi train image tao them N anh augmented
    augment_factor: int =  3,
    img_size: int = 640,
):
    """
    Offline augmentation: Tao N ban augmented cho moi anh training
    Dung khi dataset qua nho
    Neu dataset du lon -> ignore buoc nay
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    transform = get_train_transforms(img_size)
    
    for split in ["train"]:
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        
        out_img_dir.mkdir(parents=True, exist_ok = True)
        out_lbl_dir.mkdir(parents = True, exist_ok = True)
        
        img_path = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        for img_path in tqdm(img_path, desc=f"Augmenting {split}"):
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
                            ## label: class cua anh
                            ## bboxes: no nam o dau tron anh
                            class_labels.append(int(parts[0]))
                            bboxes.append([float(x) for x in parts[1:]])
                            
                            
            ## copy anh goc
            cv2.imwrite(
                str(out_img_dir / img_path.name),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            if lbl_path.exists():
                import shutil
                shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)
                
            ## create n augmented version
            for i in range(augment_factor):
                try:
                    result = transform(
                        image = image,
                        bboxes = bboxes,
                        class_labels = class_labels,
                    )
                    aug_image = result["image"]
                    aug_bboxes = result["bboxes"]
                    aug_labels = result["class_labels"]
                    
                    ## bo qua neu khon con bbox nao
                    if not aug_bboxes and bboxes:
                        continue
                    
                    aug_name = f"{img_path.stem}_aug{i}"
                    cv2.imwrite(
                        str(out_img_dir / f"{aug_name}.jpg"),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                    )
                    with open(out_lbl_dir / f"{aug_name}.txt", "w") as f:
                        for cls, bbox in zip(aug_labels, aug_bboxes):
                            coords = " ".join(f"{v:.6f}" for v in bbox)
                            f.write(f"{cls} {coords}\n")
                
                except Exception as e:
                    print(f"Augmented failed for {img_path.name}: {e}")
                    
    print("Offline augmentation complete!")
    
    
if __name__ == "__main__":
    augment_dataset_offline()
                
    