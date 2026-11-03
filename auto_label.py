"""
auto_label.py
============================================================
Автоматическая разметка видео из ground truth 3D-траектории.

Идея: у нас естьground truth (object_transform_SM_AA_track.csv) и параметры
всех 5 камер. Проецируем 3D-точку объекта на каждый кадр каждой камеры —
получаем 2D-метки (пиксель u,v) без ручной разметки.

Эти метки используются для:
  1) обучения модели детекции (YOLO) — этап обучения модели
  2) источника наблюдений для триангуляции — этап восстановления траектории

Выход:
  - data/labels_2d.csv   : frame, camera_id, u, v, in_frame, focal_mm
  - data/yolo_labels/    : один .txt файл на кадр каждой камеры
                           (формат YOLO: class x_center y_center w h)

Запуск:
    python auto_label.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from camera_calib import (
    VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMES,
    SENSOR_W_MM, SENSOR_H_MM,
    load_transform_csv, _num, build_intrinsics, build_camera_poses,
    project_point, Intrinsics, CameraPose,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

CAMERA_FILES = {
    1: "object_transform_CineCameraActor1.csv",
    2: "object_transform_CineCameraActor2.csv",
    3: "object_transform_CineCameraActor3.csv",
    4: "object_transform_CineCameraActor4.csv",
    5: "object_transform_CineCameraActor5.csv",
}

GT_FILE = "object_transform_SM_AA_track.csv"

# Размер бокса в пикселях вокруг проекции (YOLO требует бокс, а не точку).
# Подбирается эмпирически — для начала фиксированный, можно уточнить по видео.
BOX_HALF = 25  # бокс 50x50 пикселей вокруг проекции


def parse_focal_from_header(header: List[str]) -> Optional[float]:
    """Извлекает фокусное расстояние из последней колонки заголовка CSV."""
    for cell in reversed(header):
        cell = cell.strip()
        if cell and cell.isdigit():
            return float(cell)
    return None


@dataclass
class CameraSetup:
    camera_id: int
    intrinsics: Intrinsics
    poses: dict   # frame -> CameraPose


def load_all_cameras() -> List[CameraSetup]:
    setups: List[CameraSetup] = []
    for cam_id, fname in CAMERA_FILES.items():
        rows, header = load_transform_csv(str(DATA_DIR / fname))
        focal = parse_focal_from_header(header)
        if focal is None:
            raise ValueError(f"не найден фокус в заголовке {fname}: {header}")
        intrinsics = build_intrinsics(focal, SENSOR_W_MM, SENSOR_H_MM)
        poses = {p.frame: p for p in build_camera_poses(rows, intrinsics)}
        setups.append(CameraSetup(camera_id=cam_id, intrinsics=intrinsics, poses=poses))
        print(f"Камера {cam_id}: фокус={focal} мм, кадров={len(poses)}")
    return setups


def load_ground_truth() -> List[Tuple[int, np.ndarray]]:
    rows, _ = load_transform_csv(str(DATA_DIR / GT_FILE))
    points: List[Tuple[int, np.ndarray]] = []
    skipped = 0
    for row in rows:
        try:
            frame = int(row[0])
        except (ValueError, IndexError):
            skipped += 1
            continue
        vals = [_num(row[1]), _num(row[2]), _num(row[3])]
        if any(v is None for v in vals):
            skipped += 1
            continue
        point = np.array(vals, dtype=float)
        points.append((frame, point))
    if skipped:
        print(f"  [предупреждение] пропущено {skipped} строк ground truth с неразборчивыми значениями")
    return points


def yolo_line(class_id: int, u: float, v: float, w: int, h: int) -> str:
    """Одна строка в формате YOLO (координаты нормализованы 0..1)."""
    x_c = u / w
    y_c = v / h
    bw = (2 * BOX_HALF) / w
    bh = (2 * BOX_HALF) / h
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    setups = load_all_cameras()
    gt_points = load_ground_truth()
    gt_by_frame = {f: p for f, p in gt_points}

    labels_dir = DATA_DIR / "yolo_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Tuple[int, int, float, float, bool, float]] = []

    total_labeled = 0
    # Кадры видео: 0..VIDEO_FRAMES-1 (видео содержит 1001 кадр)
    for frame in range(VIDEO_FRAMES):
        if frame not in gt_by_frame:
            continue
        point = gt_by_frame[frame]
        for setup in setups:
            pose = setup.poses.get(frame)
            if pose is None:
                continue
            px = project_point(setup.intrinsics, pose, point)
            in_frame = px is not None
            if in_frame:
                u, v = px
                total_labeled += 1
                # YOLO-файл: data/yolo_labels/cam{cam_id}_frame{frame}.txt
                label_path = labels_dir / f"cam{setup.camera_id}_frame{frame}.txt"
                label_path.write_text(yolo_line(0, u, v, VIDEO_WIDTH, VIDEO_HEIGHT), encoding="utf-8")
            csv_rows.append((frame, setup.camera_id, px[0] if px else float("nan"),
                             px[1] if px else float("nan"), in_frame,
                             setup.intrinsics.focal_mm))

    # Сохраняем общий CSV с метками
    csv_path = DATA_DIR / "labels_2d.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_id", "u", "v", "in_frame", "focal_mm"])
        for row in csv_rows:
            writer.writerow(row)

    print(f"\nГотово. Размечено {total_labeled} наблюдений из {len(csv_rows)}.")
    print(f"CSV меток: {csv_path}")
    print(f"YOLO-файлы: {labels_dir} ({len(list(labels_dir.glob('*.txt')))} шт.)")


if __name__ == "__main__":
    main()
