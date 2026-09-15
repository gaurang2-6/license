"""
prepare_yolo_dataset.py
=======================
Converts the 1,501 labeled vehicle images and VOC XML annotations in
data/indian_vehicle_dataset into standard YOLO object detection format.
"""
import os
import sys
import shutil
import random
import cv2
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dataset_loader


def prepare_yolo_dataset(output_dir: str = "data/yolo_dataset", train_ratio: float = 0.85):
    abs_out = os.path.abspath(output_dir)
    print(f"[prepare_yolo] Target dataset path: {abs_out}")

    images_train_dir = os.path.join(abs_out, "images", "train")
    images_val_dir = os.path.join(abs_out, "images", "val")
    labels_train_dir = os.path.join(abs_out, "labels", "train")
    labels_val_dir = os.path.join(abs_out, "labels", "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    pairs = dataset_loader.find_dataset_pairs()
    print(f"[prepare_yolo] Found {len(pairs)} labeled pairs.")

    if not pairs:
        print("[prepare_yolo] Error: No pairs found in data/indian_vehicle_dataset.")
        return

    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    def process_split(split_pairs, img_dest, lbl_dest):
        converted_count = 0
        for pair in split_pairs:
            img_path = pair["image_path"]
            try:
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                img = None

            if img is None:
                continue

            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                continue

            base_name = f"sample_{pair['id']}"
            dst_img_path = os.path.join(img_dest, f"{base_name}.jpg")
            dst_lbl_path = os.path.join(lbl_dest, f"{base_name}.txt")

            # Copy image file
            shutil.copy2(img_path, dst_img_path)

            # Convert bounding boxes to YOLO format (0 cx cy w h)
            yolo_lines = []
            for ann in pair["annotations"]:
                if "box_coords" in ann:
                    xmin, ymin, xmax, ymax = ann["box_coords"]
                else:
                    x, y, bw, bh = ann["bbox"]
                    xmin, ymin, xmax, ymax = x, y, x + bw, y + bh

                # Clamp coordinates
                xmin = max(0, min(w - 1, xmin))
                ymin = max(0, min(h - 1, ymin))
                xmax = max(0, min(w, xmax))
                ymax = max(0, min(h, ymax))

                bw = xmax - xmin
                bh = ymax - ymin
                if bw <= 2 or bh <= 2:
                    continue

                cx = (xmin + xmax) / 2.0 / w
                cy = (ymin + ymax) / 2.0 / h
                norm_w = bw / w
                norm_h = bh / h

                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

            if yolo_lines:
                with open(dst_lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                converted_count += 1

        return converted_count

    n_train = process_split(train_pairs, images_train_dir, labels_train_dir)
    n_val = process_split(val_pairs, images_val_dir, labels_val_dir)

    yaml_content = f"""path: {abs_out.replace(os.sep, '/')}
train: images/train
val: images/val
names:
  0: license_plate
"""

    yaml_path = os.path.join(abs_out, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[prepare_yolo] Complete! Converted {n_train} train images and {n_val} val images.")
    print(f"[prepare_yolo] dataset.yaml created at {yaml_path}")


if __name__ == "__main__":
    prepare_yolo_dataset()
