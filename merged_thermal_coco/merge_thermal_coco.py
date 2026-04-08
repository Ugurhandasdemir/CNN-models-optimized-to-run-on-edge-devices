"""
Merge multiple thermal datasets into a single COCO format dataset.
Target classes: 0: Person, 1: Car, 2: OtherVehicle
"""
import json
import os
import shutil
from pathlib import Path
from PIL import Image

BASE = Path(r"c:\Users\ugurh\Downloads\Yeni klasör\_extracted")
OUTPUT = Path(r"c:\Users\ugurh\Downloads\Yeni klasör\merged_thermal_coco")

TARGET_CATEGORIES = [
    {"id": 0, "name": "Person", "supercategory": "none"},
    {"id": 1, "name": "Car", "supercategory": "none"},
    {"id": 2, "name": "OtherVehicle", "supercategory": "none"},
]

# Global counters
img_id_counter = 0
ann_id_counter = 0

# Track stats
stats = {}


def get_ids():
    global img_id_counter, ann_id_counter
    return img_id_counter, ann_id_counter


def next_img_id():
    global img_id_counter
    img_id_counter += 1
    return img_id_counter


def next_ann_id():
    global ann_id_counter
    ann_id_counter += 1
    return ann_id_counter


def process_coco_dataset(name, json_path, img_dir, class_map, split_data):
    """
    Process a COCO format dataset.
    class_map: dict mapping source category_id -> target category_id (0=Person,1=Car,2=OtherVehicle)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build image id -> image info map
    src_img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    img_anns = {}
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in class_map:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    copied = 0
    skipped = 0
    ann_count = 0

    for src_img_id, anns in img_anns.items():
        src_img = src_img_map[src_img_id]
        src_path = os.path.join(img_dir, src_img["file_name"])

        if not os.path.exists(src_path):
            skipped += 1
            continue

        new_img_id = next_img_id()
        # Create unique filename
        ext = os.path.splitext(src_img["file_name"])[1] or ".jpg"
        new_filename = f"{name}_{new_img_id:06d}{ext}"

        # Copy image
        dst_path = os.path.join(split_data["img_dir"], new_filename)
        shutil.copy2(src_path, dst_path)

        # Add image entry
        split_data["images"].append({
            "id": new_img_id,
            "file_name": new_filename,
            "width": src_img["width"],
            "height": src_img["height"],
        })

        # Add annotations
        for ann in anns:
            new_ann_id = next_ann_id()
            split_data["annotations"].append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": class_map[ann["category_id"]],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_count += 1

        copied += 1

    return copied, skipped, ann_count


def process_yolo_dataset(name, img_dir, label_dir, class_map, split_data):
    """
    Process a YOLO format dataset.
    class_map: dict mapping source YOLO class_id -> target category_id
    """
    copied = 0
    skipped = 0
    ann_count = 0

    label_files = sorted(os.listdir(label_dir))
    for lf in label_files:
        if not lf.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, lf)
        base_name = os.path.splitext(lf)[0]

        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        # Read labels and filter
        valid_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in class_map:
                        valid_lines.append((class_map[cls_id], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        if not valid_lines:
            continue

        # Get image dimensions
        with Image.open(img_path) as im:
            w, h = im.size

        new_img_id = next_img_id()
        ext = os.path.splitext(img_path)[1]
        new_filename = f"{name}_{new_img_id:06d}{ext}"

        dst_path = os.path.join(split_data["img_dir"], new_filename)
        shutil.copy2(img_path, dst_path)

        split_data["images"].append({
            "id": new_img_id,
            "file_name": new_filename,
            "width": w,
            "height": h,
        })

        for target_cls, cx, cy, bw, bh in valid_lines:
            # YOLO (cx,cy,w,h) normalized -> COCO (x,y,w,h) absolute
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw_abs = bw * w
            bh_abs = bh * h

            new_ann_id = next_ann_id()
            split_data["annotations"].append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": target_cls,
                "bbox": [round(x, 2), round(y, 2), round(bw_abs, 2), round(bh_abs, 2)],
                "area": round(bw_abs * bh_abs, 2),
                "segmentation": [],
                "iscrowd": 0,
            })
            ann_count += 1

        copied += 1

    return copied, skipped, ann_count


def process_rgbt_tiny(name, json_path, images_base, class_map, split_data):
    """
    Process RGBT-Tiny COCO annotations where images are in a separate directory.
    Image file_name in annotations looks like: DJI_0022_1/01/00000.jpg
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    src_img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image, filter by class
    img_anns = {}
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in class_map:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    copied = 0
    skipped = 0
    ann_count = 0

    for src_img_id, anns in img_anns.items():
        src_img = src_img_map[src_img_id]
        src_path = os.path.join(images_base, src_img["file_name"])

        if not os.path.exists(src_path):
            skipped += 1
            continue

        new_img_id = next_img_id()
        ext = os.path.splitext(src_img["file_name"])[1] or ".jpg"
        new_filename = f"{name}_{new_img_id:06d}{ext}"

        dst_path = os.path.join(split_data["img_dir"], new_filename)
        shutil.copy2(src_path, dst_path)

        split_data["images"].append({
            "id": new_img_id,
            "file_name": new_filename,
            "width": src_img["width"],
            "height": src_img["height"],
        })

        for ann in anns:
            new_ann_id = next_ann_id()
            split_data["annotations"].append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": class_map[ann["category_id"]],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_count += 1

        copied += 1

    return copied, skipped, ann_count


def main():
    # Create output directories
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        splits[split] = {
            "images": [],
            "annotations": [],
            "img_dir": str(img_dir),
        }

    total_stats = []

    # ========== Dataset 1: Drone Thermal Model ==========
    print("Processing: Drone Thermal Model...")
    ds_name = "drone_thermal"
    class_map = {1: 0}  # Person -> Person
    split_mapping = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in split_mapping.items():
        json_path = BASE / "Drone Thermal Model.v1i.coco" / src_split / "_annotations.coco.json"
        img_dir = BASE / "Drone Thermal Model.v1i.coco" / src_split
        if json_path.exists():
            c, s, a = process_coco_dataset(ds_name, str(json_path), str(img_dir), class_map, splits[dst_split])
            print(f"  {src_split}: {c} images, {a} annotations, {s} skipped")

    # ========== Dataset 2: thermal.v1i.coco ==========
    print("Processing: thermal.v1i.coco...")
    ds_name = "thermal_v1"
    class_map = {1: 0, 2: 2}  # person -> Person, vehicle -> OtherVehicle
    for src_split, dst_split in split_mapping.items():
        json_path = BASE / "thermal.v1i.coco" / src_split / "_annotations.coco.json"
        img_dir = BASE / "thermal.v1i.coco" / src_split
        if json_path.exists():
            c, s, a = process_coco_dataset(ds_name, str(json_path), str(img_dir), class_map, splits[dst_split])
            print(f"  {src_split}: {c} images, {a} annotations, {s} skipped")

    # ========== Dataset 3: thermal.v1i.coco (1) ==========
    print("Processing: thermal.v1i.coco (1)...")
    ds_name = "thermal_v1_alt"
    class_map = {5: 0, 2: 1, 4: 2}  # person->Person, car->Car, other vehicle->OtherVehicle
    for src_split, dst_split in split_mapping.items():
        json_path = BASE / "thermal.v1i.coco (1)" / src_split / "_annotations.coco.json"
        img_dir = BASE / "thermal.v1i.coco (1)" / src_split
        if json_path.exists():
            c, s, a = process_coco_dataset(ds_name, str(json_path), str(img_dir), class_map, splits[dst_split])
            print(f"  {src_split}: {c} images, {a} annotations, {s} skipped")

    # ========== Dataset 4: HIT-UAV (YOLO) ==========
    print("Processing: HIT-UAV...")
    ds_name = "hituav"
    class_map = {0: 0, 1: 1, 3: 2}  # Person->Person, Car->Car, OtherVehicle->OtherVehicle
    yolo_split_mapping = {"train": "train", "val": "val", "test": "test"}
    for src_split, dst_split in yolo_split_mapping.items():
        img_dir = BASE / "archive" / "hit-uav" / "images" / src_split
        label_dir = BASE / "archive" / "hit-uav" / "labels" / src_split
        if img_dir.exists() and label_dir.exists():
            c, s, a = process_yolo_dataset(ds_name, str(img_dir), str(label_dir), class_map, splits[dst_split])
            print(f"  {src_split}: {c} images, {a} annotations, {s} skipped")

    # ========== Dataset 5: RGBT-Tiny (thermal channel 01) ==========
    print("Processing: RGBT-Tiny (thermal)...")
    ds_name = "rgbt_tiny"
    class_map = {3: 0, 1: 1, 4: 2}  # pedestrian->Person, car->Car, bus->OtherVehicle
    images_base = str(BASE / "images-001")

    # Train
    json_path = BASE / "RGBT-Tiny-20260402T161330Z-3-002" / "RGBT-Tiny" / "annotations_coco" / "instances_01_train2017.json"
    if json_path.exists():
        c, s, a = process_rgbt_tiny(ds_name, str(json_path), images_base, class_map, splits["train"])
        print(f"  train: {c} images, {a} annotations, {s} skipped")

    # Test
    json_path = BASE / "RGBT-Tiny-20260402T161330Z-3-002" / "RGBT-Tiny" / "annotations_coco" / "instances_01_test2017.json"
    if json_path.exists():
        c, s, a = process_rgbt_tiny(ds_name, str(json_path), images_base, class_map, splits["test"])
        print(f"  test: {c} images, {a} annotations, {s} skipped")

    # ========== Save COCO JSONs ==========
    print("\nSaving COCO annotations...")
    for split_name, split_data in splits.items():
        coco_output = {
            "images": split_data["images"],
            "annotations": split_data["annotations"],
            "categories": TARGET_CATEGORIES,
        }
        out_path = OUTPUT / split_name / "_annotations.coco.json"
        with open(out_path, "w") as f:
            json.dump(coco_output, f)
        print(f"  {split_name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")

    # ========== Summary ==========
    print("\n========== SUMMARY ==========")
    total_imgs = sum(len(s["images"]) for s in splits.values())
    total_anns = sum(len(s["annotations"]) for s in splits.values())
    print(f"Total images: {total_imgs}")
    print(f"Total annotations: {total_anns}")

    # Count per class
    class_counts = {0: 0, 1: 0, 2: 0}
    for s in splits.values():
        for ann in s["annotations"]:
            class_counts[ann["category_id"]] += 1
    print(f"  Person: {class_counts[0]}")
    print(f"  Car: {class_counts[1]}")
    print(f"  OtherVehicle: {class_counts[2]}")
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
