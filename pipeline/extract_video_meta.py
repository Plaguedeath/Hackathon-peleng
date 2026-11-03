import csv
import json
import os
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "runs" / "metadata"

VIDEO_FILES = sorted(DATA_DIR.glob("Way1.FinalImage.CineCameraActor*.mp4"))
CAMERA_IDS = [1, 2, 3, 4, 5]


def extract_meta(video_path, cam_id):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    meta = {
        "camera_id": cam_id,
        "file": video_path.name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()
    return meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, v in enumerate(VIDEO_FILES):
        cam_id = CAMERA_IDS[i] if i < len(CAMERA_IDS) else i + 1
        meta = extract_meta(v, cam_id)
        metas.append(meta)
        print(f"Cam{cam_id}: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps, {meta['frame_count']} frames")

    # Check consistency
    fps_set = {m["fps"] for m in metas}
    size_set = {(m["width"], m["height"]) for m in metas}
    count_set = {m["frame_count"] for m in metas}
    print(f"\nFPS values: {fps_set}")
    print(f"Resolutions: {size_set}")
    print(f"Frame counts: {count_set}")

    with open(OUT_DIR / "video_meta.json", "w") as f:
        json.dump(metas, f, indent=2)

    # Resolution summary for calibration
    if len(size_set) == 1:
        w, h = next(iter(size_set))
        print(f"\nAll videos: {w}x{h}. Use: VIDEO_W={w} VIDEO_H={h} python3 camera_calibration.py")
    else:
        print("\nWARNING: Different resolutions across videos! Calibration must be per-camera.")

    return metas


if __name__ == "__main__":
    main()
