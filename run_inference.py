"""
run_inference.py
============================================================
Этап 3: Инференс обученной YOLO модели на всех 5 видео.

Загружает best.pt, прогоняет каждое видео кадр за кадром,
сохраняет детекции в CSV формат для триангуляции.

Выход:
  - runs/detections/cam{N}_detections.csv
  - runs/detections/all_detections.csv
  - (опционально) runs/detections/cam{N}_annotated.mp4
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
MODEL_DIR = RUNS_DIR / "peleng_detect" / "weights"
DETECTIONS_DIR = RUNS_DIR / "detections"

VIDEO_FILES = {
    1: DATA_DIR / "Way1.FinalImage.CineCameraActor1.mp4",
    2: DATA_DIR / "Way1.FinalImage.CineCameraActor2.mp4",
    3: DATA_DIR / "Way1.FinalImage.CineCameraActor3.mp4",
    4: DATA_DIR / "Way1.FinalImage.CineCameraActor4.mp4",
    5: DATA_DIR / "Way1.FinalImage.CineCameraActor5.mp4",
}

CLASS_ID_OBJECT = 0  # класс объекта (только один класс)
DEFAULT_CONF = 0.25


def load_model(weights_path: Path | None = None) -> YOLO:
    """Загружает обученную модель YOLO."""
    if weights_path is None:
        weights_path = MODEL_DIR / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Веса не найдены: {weights_path}. Сначала запустите train_yolo.py"
        )
    print(f"Загрузка модели: {weights_path}")
    return YOLO(str(weights_path))


def infer_video(
    cam_id: int,
    video_path: Path,
    model: YOLO,
    conf: float = DEFAULT_CONF,
    out_dir: Path = DETECTIONS_DIR,
    draw: bool = False,
) -> list[dict]:
    """Прогоняет одно видео через модель, возвращает список детекций."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    print(f"  Камера {cam_id}: {w}x{h} @ {fps:.1f} fps, {total_frames} кадров")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != CLASS_ID_OBJECT:
                    continue
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                rows.append({
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
                    cv2.putText(frame, f"obj {conf_val:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if draw:
            writer.write(frame)

        idx += 1
        if idx % 200 == 0:
            print(f"    Кадр {idx}/{total_frames}...")

    cap.release()
    if writer:
        writer.release()

    # Сохраняем CSV для этой камеры
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "camera_id", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
        for d in rows:
            writer_csv.writerow([
                d["frame"], d["camera_id"], d["x1"], d["y1"], d["x2"], d["y2"],
                d["cx"], d["cy"], d["confidence"],
            ])

    print(f"  Камера {cam_id}: {len(rows)} детекций за {idx} кадров -> {csv_path}")
    return rows


def run_all(
    weights_path: Path | None = None,
    conf: float = DEFAULT_CONF,
    draw: bool = False,
) -> list[dict]:
    """Запускает инференс на всех 5 видео."""
    model = load_model(weights_path)
    all_rows = []

    for cam_id, video_path in VIDEO_FILES.items():
        rows = infer_video(cam_id, video_path, model, conf=conf, draw=draw)
        all_rows.extend(rows)

    # Объединённый файл всех детекций
    combined_path = DETECTIONS_DIR / "all_detections.csv"
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_id", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
        for d in all_rows:
            writer.writerow([
                d["frame"], d["camera_id"], d["x1"], d["y1"], d["x2"], d["y2"],
                d["cx"], d["cy"], d["confidence"],
            ])

    print(f"\nВсего детекций: {len(all_rows)} -> {combined_path}")
    return all_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO inference on 5 camera videos")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--draw", action="store_true", help="Draw boxes and save annotated videos")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else None
    run_all(weights_path=weights, conf=args.conf, draw=args.draw)