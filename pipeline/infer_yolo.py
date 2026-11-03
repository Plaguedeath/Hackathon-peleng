import csv
import json
import os
from pathlib import Path

import cv2

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DATA_DIR = Path(__file__).resolve().parent / "data"
CALIB_DIR = RUNS_DIR / "calibration"
MODEL_DIR = RUNS_DIR / "model"
DETECTIONS_DIR = RUNS_DIR / "detections"
RESULTS_DIR = RUNS_DIR / "results"

VIDEO_FILES = {
    1: DATA_DIR / "Way1.FinalImage.CineCameraActor1.mp4",
    2: DATA_DIR / "Way1.FinalImage.CineCameraActor2.mp4",
    3: DATA_DIR / "Way1.FinalImage.CineCameraActor3.mp4",
    4: DATA_DIR / "Way1.FinalImage.CineCameraActor4.mp4",
    5: DATA_DIR / "Way1.FinalImage.CineCameraActor5.mp4",
}

CLASS_ID_TRUCK = 0  # YOLO class index after training on single class


def load_model(weights_path=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics opencv-python")

    if weights_path is None:
        weights_path = MODEL_DIR / "truck_detector" / "weights" / "best.pt"
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}. Train first.")
    return YOLO(str(weights_path))


def infer_video(cam_id, video_path, model, conf=0.25, out_dir=DETECTIONS_DIR, draw=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_path = out_dir / f"cam{cam_id}_detections.csv"
    rows = []
    idx = 0

    writer = None
    if draw:
        writer = cv2.VideoWriter(
            str(out_dir / f"cam{cam_id}_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != CLASS_ID_TRUCK:
                    continue
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "frame": idx,
                    "camera_id": cam_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "confidence": conf_val,
                })
                if draw:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"truck {conf_val:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rows.extend(detections)
        if draw:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer:
        writer.release()

    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "camera_id", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
        for d in rows:
            writer_csv.writerow([
                d["frame"], d["camera_id"], d["x1"], d["y1"], d["x2"], d["y2"],
                d["cx"], d["cy"], d["confidence"],
            ])

    print(f"Cam{cam_id}: {len(rows)} detections across {idx} frames -> {csv_path}")
    return rows


def infer_all(weights_path=None, conf=0.25, draw=False):
    model = load_model(weights_path)
    all_rows = []
    for cam_id, video_path in VIDEO_FILES.items():
        rows = infer_video(cam_id, video_path, model, conf=conf, draw=draw)
        all_rows.extend(rows)

    # Combined detections file
    combined = DETECTIONS_DIR / "all_detections.csv"
    with open(combined, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_id", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
        for d in all_rows:
            writer.writerow([
                d["frame"], d["camera_id"], d["x1"], d["y1"], d["x2"], d["y2"],
                d["cx"], d["cy"], d["confidence"],
            ])
    print(f"\nCombined detections: {len(all_rows)} -> {combined}")
    return all_rows


if __name__ == "__main__":
    infer_all()
