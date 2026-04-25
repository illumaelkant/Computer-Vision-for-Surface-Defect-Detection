"""
Convert pascal VOC XML annotations sang yolo format (.txt)
va split thanh train/val/test
"""

import os

import xml.etree.ElementTree as ET
import shutil

import random
from pathlib import Path
from tqdm import tqdm
import yaml


CLASS_MAP = {
    "crazing" : 0,
    "inclusion" : 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}

split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}



def parse_voc_xml(xml_path: str) -> list[dict]:
    '''
    Parse 1 file XML DOC, tra ve list cac object annotation
    '''
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
        x_min = float(bndbox.find("xmin").text)
        y_min = float(bndbox.find("ymin").text)
        x_max = float(bndbox.find("xmax").text)
        y_max = float(bndbox.find("ymax").text)
        
        ## convert sang yolo format
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min)/img_width
        height = (y_max - y_min)/img_height
        
        ## clip de tranh gia tri out-of-bounf do annotaion loi
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


    