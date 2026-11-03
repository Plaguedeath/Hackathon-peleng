import csv
import os
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
DATASET_DIR = RUNS_DIR / "dataset"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"

VIDEO_FILES = {
    1: DATA_DIR / "Way1.FinalImage.CineCameraActor1.mp4",
    2: DATA_DIR / "Way1.FinalImage.CineCameraActor2.mp4",
    3: DATA_DIR / "Way1.FinalImage.CineCameraActor3.mp4",
    4: DATA_DIR / "Way1.FinalImage.CineCameraActor4.mp4",
    5: DATA_DIR / "Way1.FinalImage.CineCameraActor5.mp4",
}

LABEL_FILES = {
    "train": DATASET_DIR / "labels" / "train",
    "val": DATASET_DIR / "labels" / "val",
}


def load_label_frames(split):
    frames_by_cam = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()}
    lbl_dir = LABEL_FILES[split]
    if not lbl_dir.exists():
        return frames_by_cam
    for lbl in lbl_dir.glob("*.txt"):
        # filename: cam{cam_id}_frame{frame:06d}.txt
        stem = lbl.stem
        parts = stem.split("_")
        cam_id = int(parts[0].replace("cam", ""))
        frame = int(parts[1].replace("frame", ""))
        frames_by_cam[cam_id].add(frame)
    return frames_by_cam


def extract_frames_for_split(split):
    img_dir = TRAIN_IMAGES if split == "train" else VAL_IMAGES
    img_dir.mkdir(parents=True, exist_ok=True)
    frames_needed = load_label_frames(split)

    total_copied = 0
    for cam_id, video_path in VIDEO_FILES.items():
        needed = sorted(frames_needed.get(cam_id, set()))
        if not needed:
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open {video_path}")
            continue
        needed_set = set(needed)
        max_frame = max(needed)
        idx = 0
        copied = 0
        while idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in needed_set:
                out_name = f"cam{cam_id}_frame{idx:06d}.jpg"
                cv2.imwrite(str(img_dir / out_name), frame)
                copied += 1
            idx += 1
        cap.release()
        total_copied += copied
        print(f"Cam{cam_id} [{split}]: copied {copied} frames")

    print(f"Total {split} images: {total_copied}")


def main():
    for split in ("train", "val"):
        extract_frames_for_split(split)

    # Verify image/label pairing
    for split in ("train", "val"):
        img_dir = TRAIN_IMAGES if split == "train" else VAL_IMAGES
        lbl_dir = LABEL_FILES[split]
        imgs = {p.stem for p in img_dir.glob("*.jpg")}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        missing_labels = imgs - lbls
        missing_images = lbls - imgs
        print(f"\n[{split}] images={len(imgs)} labels={len(lbls)} "
              f"missing_labels={len(missing_labels)} missing_images={len(missing_images)}")


if __name__ == "__main__":
    main()
