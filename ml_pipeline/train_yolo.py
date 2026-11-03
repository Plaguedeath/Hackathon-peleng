import csv
import os
import random
import shutil
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
CALIB_DIR = RUNS_DIR / "calibration"
DATASET_DIR = RUNS_DIR / "dataset"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_LABELS = DATASET_DIR / "labels" / "val"
MODEL_DIR = RUNS_DIR / "model"

VIDEO_W = int(os.environ.get("VIDEO_W", "1920"))
VIDEO_H = int(os.environ.get("VIDEO_H", "1080"))

BOX_SIZE_PX = 80  # half-size of auto-generated bbox around projected point
TRAIN_RATIO = 0.8


def load_projections(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_yolo_label(u, v, w_px, h_px, box_size=BOX_SIZE_PX):
    x1 = max(0, u - box_size)
    y1 = max(0, v - box_size)
    x2 = min(w_px, u + box_size)
    y2 = min(h_px, v + box_size)
    bw = x2 - x1
    bh = y2 - y1
    xc = (x1 + x2) / 2 / w_px
    yc = (y1 + y2) / 2 / h_px
    wn = bw / w_px
    hn = bh / h_px
    return xc, yc, wn, hn


def build_dataset(projections_path, output_dir=DATASET_DIR):
    rows = load_projections(projections_path)
    visible = [r for r in rows if r["visible"].strip() == "True"]

    # Group by camera+frame for image naming
    # Each projection row corresponds to a frame+camera; we need the actual video frame image
    # We will extract frames on demand during training prep or assume frames already extracted.
    # Here we create label files keyed by camera_id_frame.
    label_entries = []
    for r in visible:
        cam_id = int(r["camera_id"])
        frame = int(r["frame"])
        u = float(r["u"])
        v = float(r["v"])
        xc, yc, wn, hn = make_yolo_label(u, v, VIDEO_W, VIDEO_H)
        label_entries.append({
            "cam_id": cam_id,
            "frame": frame,
            "label": f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n",
        })

    print(f"Total visible projections (labels): {len(label_entries)}")
    return label_entries


def write_yolo_dataset(label_entries, val_ratio=0.2):
    for d in (TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS):
        d.mkdir(parents=True, exist_ok=True)

    random.shuffle(label_entries)
    n_val = int(len(label_entries) * val_ratio)
    val_entries = label_entries[:n_val]
    train_entries = label_entries[n_val:]

    def write_split(entries, img_dir, lbl_dir):
        for e in entries:
            cam_id = e["cam_id"]
            frame = e["frame"]
            # Source frame image path - will be populated by frame extraction step
            # We create a placeholder; actual image copying happens in extract_frames_for_training
            label_name = f"cam{cam_id}_frame{frame:06d}.txt"
            label_path = lbl_dir / label_name
            with open(label_path, "w") as f:
                f.write(e["label"])

    write_split(train_entries, TRAIN_IMAGES, TRAIN_LABELS)
    write_split(val_entries, VAL_IMAGES, VAL_LABELS)

    print(f"Train labels: {len(train_entries)}, Val labels: {len(val_entries)}")
    return train_entries, val_entries


def create_data_yaml(path, nc=1, names=["truck"]):
    data = {
        "train": str(TRAIN_IMAGES),
        "val": str(VAL_IMAGES),
        "nc": nc,
        "names": names,
    }
    with open(path, "w") as f:
        yaml.dump(data, f)
    print(f"data.yaml written: {path}")


def train_yolo(data_yaml, epochs=50, imgsz=640, batch=8, pretrained="yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        return None

    model = YOLO(pretrained)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(MODEL_DIR),
        name="truck_detector",
        exist_ok=True,
        verbose=True,
    )
    best_weights = MODEL_DIR / "truck_detector" / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")
    return best_weights


def main():
    projections_path = CALIB_DIR / "projections.csv"
    if not projections_path.exists():
        print(f"Projections not found: {projections_path}. Run camera_calibration.py first.")
        return

    label_entries = build_dataset(projections_path)

    # We need actual frame images for training. This script creates label files;
    # frame extraction and image copying is a separate step (see extract_frames_for_training).
    train_entries, val_entries = write_yolo_dataset(label_entries)

    data_yaml = RUNS_DIR / "data.yaml"
    create_data_yaml(data_yaml)

    print("\nDataset prepared. Next steps:")
    print("1. Extract actual frame images from videos and copy to dataset/images/{train,val}/")
    print("2. Run: python3 train_yolo.py --train  (once ultralytics is installed)")
    print("3. Best weights will be at runs/model/truck_detector/weights/best.pt")


if __name__ == "__main__":
    main()
