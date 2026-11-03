"""
triangulate.py
============================================================
Этап 4: Триангуляция 3D-траектории из 2D-детекций.

Читает all_detections.csv (результат run_inference.py) и параметры камер
из camera_calib.py. Для каждого кадра собирает наблюдения ≥2 камер,
строит лучи обзора и пересекает их методом наименьших квадратов (DLT).

Выход:
  - runs/trajectory/trajectory_raw.csv  (сырая траектория)
  - runs/trajectory/triangulation_log.csv  (подробности по кадрам)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from camera_calib import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    load_transform_csv, _num, build_intrinsics, build_camera_poses,
    project_point, Intrinsics, CameraPose,
    ray_from_pixel, triangulate_point as calib_triangulate_point,
)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
DETECTIONS_DIR = RUNS_DIR / "detections"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"
DATA_DIR = ROOT / "data"

CAMERA_FILES = {
    1: "object_transform_CineCameraActor1.csv",
    2: "object_transform_CineCameraActor2.csv",
    3: "object_transform_CineCameraActor3.csv",
    4: "object_transform_CineCameraActor4.csv",
    5: "object_transform_CineCameraActor5.csv",
}

DETECTIONS_FILE = DETECTIONS_DIR / "all_detections.csv"
OUTPUT_RAW = TRAJECTORY_DIR / "trajectory_raw.csv"
OUTPUT_LOG = TRAJECTORY_DIR / "triangulation_log.csv"


def load_all_cameras() -> dict[int, tuple[Intrinsics, dict[int, CameraPose]]]:
    """Загружает интринсики и покадровые позы для всех 5 камер."""
    cameras = {}
    for cam_id, fname in CAMERA_FILES.items():
        rows, header = load_transform_csv(str(DATA_DIR / fname))
        focal = None
        for cell in reversed(header):
            cell = cell.strip()
            if cell and cell.replace('.', '').isdigit():
                focal = float(cell)
                break
        if focal is None:
            raise ValueError(f"Не найден фокус в заголовке {fname}")
        intrinsics = build_intrinsics(focal)
        poses = {p.frame: p for p in build_camera_poses(rows, intrinsics)}
        cameras[cam_id] = (intrinsics, poses)
        print(f"  Камера {cam_id}: фокус={focal} мм, кадров={len(poses)}")
    return cameras


def load_detections(path: Path) -> dict[int, list[dict]]:
    """Читает all_detections.csv, группирует по кадрам."""
    detections: dict[int, list[dict]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            detections.setdefault(frame, []).append({
                "camera_id": int(row["camera_id"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "confidence": float(row["confidence"]),
            })
    return detections


def triangulate_frame(
    frame: int,
    observations: list[dict],
    cameras: dict[int, tuple[Intrinsics, dict[int, CameraPose]]],
) -> tuple[Optional[np.ndarray], int]:
    """
    Триангулирует одну 3D-точку для кадра.
    Возвращает (X_world, num_cameras_used) или (None, num_cameras).
    """
    valid_obs = []
    for obs in observations:
        cam_id = obs["camera_id"]
        if cam_id not in cameras:
            continue
        intrinsics, poses = cameras[cam_id]
        pose = poses.get(frame)
        if pose is None:
            continue
        valid_obs.append((pose, intrinsics, obs["cx"], obs["cy"]))

    if len(valid_obs) < 2:
        return None, len(valid_obs)

    # Используем функцию из camera_calib (DLT с весами)
    X = calib_triangulate_point(valid_obs)
    return X, len(valid_obs)


def main() -> None:
    if not DETECTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Файл детекций не найден: {DETECTIONS_FILE}. "
            f"Сначала запустите run_inference.py"
        )

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка камер...")
    cameras = load_all_cameras()

    print("Загрузка детекций...")
    detections = load_detections(DETECTIONS_FILE)
    print(f"  Кадров с детекциями: {len(detections)}")

    all_frames = sorted(detections.keys())
    results = []
    log_rows = []

    print("Триангуляция...")
    for frame in all_frames:
        obs = detections[frame]
        X, n_used = triangulate_frame(frame, obs, cameras)

        if X is not None:
            results.append({
                "frame": frame,
                "x": float(X[0]),
                "y": float(X[1]),
                "z": float(X[2]),
                "n_cameras": n_used,
                "triangulated": True,
            })
            log_rows.append({
                "frame": frame,
                "status": "OK",
                "n_obs": len(obs),
                "n_used": n_used,
                "x": X[0], "y": X[1], "z": X[2],
            })
        else:
            results.append({
                "frame": frame,
                "x": None, "y": None, "z": None,
                "n_cameras": n_used,
                "triangulated": False,
            })
            log_rows.append({
                "frame": frame,
                "status": "INSUFFICIENT_CAMERAS",
                "n_obs": len(obs),
                "n_used": n_used,
                "x": "", "y": "", "z": "",
            })

    # Сохраняем сырую траекторию
    with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "n_cameras", "triangulated"])
        for r in results:
            writer.writerow([
                r["frame"],
                r["x"] if r["x"] is not None else "",
                r["y"] if r["y"] is not None else "",
                r["z"] if r["z"] is not None else "",
                r["n_cameras"],
                r["triangulated"],
            ])

    # Сохраняем подробный лог
    with open(OUTPUT_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "status", "n_obs", "n_used", "x", "y", "z"])
        for r in log_rows:
            writer.writerow([r["frame"], r["status"], r["n_obs"], r["n_used"], r["x"], r["y"], r["z"]])

    n_tri = sum(1 for r in results if r["triangulated"])
    print(f"\nГотово: триангулировано {n_tri}/{len(results)} кадров")
    print(f"  Сырая траектория: {OUTPUT_RAW}")
    print(f"  Лог: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()