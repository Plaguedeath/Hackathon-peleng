"""
camera_calib.py
============================================================
Калібровка камер из CSV-трансформаций Unreal Engine.

Построение матриц проецирования P = K·[R|t] для двух задач:
  1) авторазметка 3D -> 2D (готовит обучающую выборку для модели детекции)
  2) триангуляция 2D -> 3D (восстановление траектории объекта)

Конвенция камеры (подобрана экспериментально против ground truth):
    forward = столбец 0 матрицы поворота UE   (оптическая ось)
    right   = столбец 1                       (вправо в кадре)
    down    = normalize(cross(right, forward)) (вниз в кадре)
    world->камера: строки [right, down, forward]

Запуск:
    python camera_calib.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Параметры видео (измерены через OpenCV, все 5 камер идентичны)
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30.0
VIDEO_FRAMES = 1001      # кадров в видео (0..1000)
CSV_FRAMES = 1002        # строк данных в CSV (кадры 0..1001) — на 1 больше

SENSOR_W_MM = 23.76      # размер сенсора из заголовка CSV
SENSOR_H_MM = 13.365


@dataclass
class Intrinsics:
    K: np.ndarray        # 3x3 матрица камеры
    width: int
    height: int
    focal_mm: float
    sensor_w_mm: float
    sensor_h_mm: float


@dataclass
class CameraPose:
    frame: int
    loc: np.ndarray      # позиция камеры в мире (UE, см), форма (3,)
    rot: np.ndarray      # поворот (roll, pitch, yaw) в градусах UE, форма (3,)
    R: np.ndarray        # матрица поворота мир<-локально UE 3x3
    W2C: np.ndarray      # поворот мир->камера 3x3 (строки: right, down, forward)
    t: np.ndarray        # позиция камеры в мире, форма (3,1)


# ===========================================================================
# Загрузка CSV
# ===========================================================================

def load_transform_csv(path: str, encoding: str = "cp1251") -> Tuple[List[List[str]], List[str]]:
    """Читает CSV-трансформацию (разделитель ';', кодировка cp1251)."""
    rows: List[List[str]] = []
    header: List[str] = []
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, [])
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            rows.append(row)
    return rows, header


def _num(value: str) -> Optional[float]:
    """Парсит число, заменяя запятую на точку. Возвращает None при ошибке."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except ValueError:
        return None


# ===========================================================================
# Интринсики камеры
# ===========================================================================

def build_intrinsics(
    focal_mm: float,
    sensor_w_mm: float = SENSOR_W_MM,
    sensor_h_mm: float = SENSOR_H_MM,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> Intrinsics:
    """Матрица внутренних параметров K."""
    fx = focal_mm * width / sensor_w_mm
    fy = focal_mm * height / sensor_h_mm
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return Intrinsics(K=K, width=width, height=height,
                      focal_mm=focal_mm, sensor_w_mm=sensor_w_mm,
                      sensor_h_mm=sensor_h_mm)


# ===========================================================================
# Экстринсики
# ===========================================================================

def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) — типично для Unreal Engine."""
    r, p, y = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr,  cr]], dtype=float)
    Ry = np.array([[cp,  0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def build_camera_poses(
    rows: List[List[str]],
    intrinsics: Intrinsics,
) -> List[CameraPose]:
    """Собирает покадровые экстринсики для одной камеры."""
    poses: List[CameraPose] = []
    skipped = 0
    for row in rows:
        try:
            frame = int(row[0])
        except (ValueError, IndexError):
            skipped += 1
            continue
        loc_vals = [_num(row[1]), _num(row[2]), _num(row[3])]
        rot_vals = [
            _num(row[4]) if len(row) > 4 else 0.0,
            _num(row[5]) if len(row) > 5 else 0.0,
            _num(row[6]) if len(row) > 6 else 0.0,
        ]
        if any(v is None for v in loc_vals) or any(v is None for v in rot_vals):
            skipped += 1
            continue
        loc = np.array(loc_vals, dtype=float)
        rot = np.array(rot_vals, dtype=float)
        R = _rotation_matrix(rot[0], rot[1], rot[2])
        forward = R[:, 0].copy()   # оптическая ось камеры в мире
        right = R[:, 1].copy()     # вправо в кадре
        down = np.cross(right, forward)
        down = down / np.linalg.norm(down)
        # строки world->камера: [right, down, forward]
        W2C = np.stack([right, down, forward], axis=0)
        t = loc.reshape(3, 1)
        poses.append(CameraPose(frame=frame, loc=loc, rot=rot, R=R,
                                W2C=W2C, t=t))
    if skipped:
        print(f"  [предупреждение] пропущено {skipped} строк с неразборчивыми значениями")
    return poses


# ===========================================================================
# Проецирование 3D -> 2D (авторазметка обучающей выборки)
# ===========================================================================

def project_point(
    intrinsics: Intrinsics,
    pose: CameraPose,
    point_world: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Проецирует мировую точку (UE, см) на пиксель (u, v) или None."""
    point_world = np.asarray(point_world, dtype=float).reshape(3, 1)
    X_cam = (pose.W2C @ (point_world - pose.t)).flatten()

    if X_cam[2] <= 0:
        return None  # точка за камерой

    u = intrinsics.K[0, 0] * X_cam[0] / X_cam[2] + intrinsics.K[0, 2]
    v = intrinsics.K[1, 1] * X_cam[1] / X_cam[2] + intrinsics.K[1, 2]

    if not (0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height):
        return None  # вне кадра

    return (float(u), float(v))


# ===========================================================================
# Луч обзора 2D -> 3D (триангуляция)
# ===========================================================================

def ray_from_pixel(
    intrinsics: Intrinsics,
    pose: CameraPose,
    u: float,
    v: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Источник и направление луча в мировых координатах для пикселя (u, v)."""
    kinv = np.linalg.inv(intrinsics.K)
    pixel = np.array([u, v, 1.0], dtype=float)
    ray_cam = kinv @ pixel
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_world = pose.W2C.T @ ray_cam   # камера->мир
    origin = pose.loc.copy()
    return origin, ray_world


def triangulate_point(
    observations: List[Tuple[CameraPose, Intrinsics, float, float]],
) -> Optional[np.ndarray]:
    """
    Линейная триангуляции точки по наблюдениям из >= 2 камер.
    observations: список (pose, intrinsics, u, v).
    """
    if len(observations) < 2:
        return None

    origins, dirs = [], []
    for pose, intrinsics, u, v in observations:
        o, d = ray_from_pixel(intrinsics, pose, u, v)
        origins.append(o)
        dirs.append(d / np.linalg.norm(d))

    origins = np.stack(origins)
    dirs = np.stack(dirs)

    # Для каждого луча: P_i = I - d_i d_i^T  (проекция на плоскость перпендикулярно лучу)
    # Условие пересечения: P_i @ X = P_i @ o_i  для всех i
    A_blocks, b_blocks = [], []
    for o, d in zip(origins, dirs):
        d_col = d.reshape(3, 1)
        P = np.eye(3) - (d_col @ d_col.T)
        A_blocks.append(P)
        b_blocks.append(P @ o.reshape(3, 1))

    A_mat = np.vstack(A_blocks)
    b_mat = np.vstack(b_blocks)
    X, _, _, _ = np.linalg.lstsq(A_mat, b_mat, rcond=None)
    return X.flatten()


# ===========================================================================
# Самопроверка
# ===========================================================================

def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    cam_rows, cam_header = load_transform_csv(
        str(data_dir / "object_transform_CineCameraActor1.csv")
    )
    print(f"Камера 1: строк данных={len(cam_rows)}")
    intrinsics = build_intrinsics(35.0, SENSOR_W_MM, SENSOR_H_MM)
    poses = {p.frame: p for p in build_camera_poses(cam_rows, intrinsics)}

    gt_rows, gt_header = load_transform_csv(
        str(data_dir / "object_transform_SM_AA_track.csv")
    )
    print(f"Ground truth: строк данных={len(gt_rows)}")

    print("\n=== Самопроверка проекции 3D -> 2D (Камера 1, первые 10 кадров) ===")
    in_frame = 0
    for row in gt_rows[:10]:
        frame = int(row[0])
        point = np.array([_num(row[1]), _num(row[2]), _num(row[3])], dtype=float)
        pose = poses.get(frame)
        if pose is None:
            print(f"кадр {frame}: нет данных камеры")
            continue
        px = project_point(intrinsics, pose, point)
        status = "OK" if px is not None else "за кадром/за камерой"
        if px is not None:
            in_frame += 1
        print(f"кадр {frame:3d}: мир=({point[0]:8.1f},{point[1]:8.1f},{point[2]:6.1f}) "
              f"-> пиксель={px}  [{status}]")

    print(f"\nВ кадр попали {in_frame} из 10 точек (Камера 1).")
    if in_frame == 0:
        print("Требуется донастройка конвенции осей.")


if __name__ == "__main__":
    main()
