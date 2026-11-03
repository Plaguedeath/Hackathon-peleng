"""
postprocess.py
============================================================
Этап 5: Постобработка 3D-траектории.

- Удаление выбросов (межкадровые скачки > порога)
- Сглаживание фильтром Савицкого-Голея (сохраняет форму траектории)
- Интерполяция пропусков (кадры, где триангуляция не удалась)
- Опционально: фильтр Калмана для динамического сглаживания

Выход:
  - runs/trajectory/trajectory_smooth.csv  (очищенная траектория)
  - runs/trajectory/postprocess_log.csv    (что делалось на каждом кадре)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"

INPUT_RAW = TRAJECTORY_DIR / "trajectory_raw.csv"
OUTPUT_SMOOTH = TRAJECTORY_DIR / "trajectory_smooth.csv"
OUTPUT_LOG = TRAJECTORY_DIR / "postprocess_log.csv"

# Параметры постобработки
MAX_FRAME_JUMP_M = 5.0      # максимальный прыжок между кадрами (м) — выше = выброс
SG_WINDOW = 11              # окно Савицкого-Голея (нечётное, ~0.3-0.5 сек при 30 fps)
SG_POLYORDER = 3            # порядок полинома
KALMAN_PROCESS_NOISE = 0.1  # шум процесса для Калмана
KALMAN_MEAS_NOISE = 1.0     # шум измерения для Калмана


def load_trajectory(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Читает trajectory_raw.csv.
    Возвращает: (frames, points, mask)
    - frames: список номеров кадров
    - points: массив (N, 3) с NaN для пропусков
    - mask: булев массив (N,) — True где есть триангуляция
    """
    frames = []
    points = []
    mask = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            frames.append(frame)
            if row["x"] == "" or row["x"] is None:
                points.append([np.nan, np.nan, np.nan])
                mask.append(False)
            else:
                points.append([float(row["x"]), float(row["y"]), float(row["z"])])
                mask.append(row["triangulated"] == "True")

    return frames, np.array(points, dtype=float), np.array(mask, dtype=bool)


def remove_outliers(frames: list[int], points: np.ndarray, mask: np.ndarray, max_jump: float = MAX_FRAME_JUMP_M) -> np.ndarray:
    """Помечает выбросы как NaN на основе межкадровой скорости."""
    cleaned = points.copy()
    for i in range(1, len(points)):
        if not mask[i] or not mask[i-1]:
            continue
        dist = np.linalg.norm(points[i] - points[i-1])
        if dist > max_jump:
            cleaned[i] = [np.nan, np.nan, np.nan]
            mask[i] = False
    return cleaned


def interpolate_gaps(frames: list[int], points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Интерполирует пропуски (NaN) линейно по времени."""
    filled = points.copy()
    for axis in range(3):
        valid = mask & ~np.isnan(points[:, axis])
        if valid.sum() < 2:
            continue
        f = interp1d(
            np.array(frames)[valid],
            points[valid, axis],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        missing = ~valid
        filled[missing, axis] = f(np.array(frames)[missing])
    return filled


def savgol_smooth(frames: list[int], points: np.ndarray, mask: np.ndarray,
                  window: int = SG_WINDOW, polyorder: int = SG_POLYORDER) -> np.ndarray:
    """Сглаживает каждую координату фильтром Савицкого-Голея."""
    smoothed = points.copy()
    n = len(points)
    if n < window:
        window = n if n % 2 == 1 else n - 1
    if window < 3:
        return smoothed

    for axis in range(3):
        # Сглаживаем только где есть данные (после интерполяции — везде)
        smoothed[:, axis] = savgol_filter(points[:, axis], window, polyorder)
    return smoothed


class KalmanFilter1D:
    """Простой 1D фильтр Калмана для постоянной скорости."""
    def __init__(self, process_noise: float, meas_noise: float, dt: float = 1/30):
        self.dt = dt
        self.x = np.array([0.0, 0.0])  # [position, velocity]
        self.P = np.eye(2) * 1000.0
        self.Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * process_noise
        self.R = np.array([[meas_noise]])
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])

    def init(self, pos: float, vel: float = 0.0):
        self.x = np.array([pos, vel])
        self.P = np.eye(2) * 1000.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = (self.P @ self.H.T) / S[0, 0]
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0])


def kalman_smooth(frames: list[int], points: np.ndarray, mask: np.ndarray,
                  process_noise: float = KALMAN_PROCESS_NOISE,
                  meas_noise: float = KALMAN_MEAS_NOISE) -> np.ndarray:
    """Применяет фильтр Калмана к каждой координате (forward-backward)."""
    smoothed = points.copy()
    dt = (frames[1] - frames[0]) / 30.0 if len(frames) > 1 else 1/30

    for axis in range(3):
        # Forward pass
        kf = KalmanFilter1D(process_noise, meas_noise, dt)
        first_valid = np.where(mask)[0]
        if len(first_valid) == 0:
            continue
        kf.init(points[first_valid[0], axis])

        forward_est = np.zeros(len(points))
        for i in range(len(points)):
            kf.predict()
            if mask[i]:
                forward_est[i] = kf.update(points[i, axis])
            else:
                forward_est[i] = kf.x[0]

        # Backward pass (Rauch-Tung-Striebel smoother упрощённый)
        backward_est = forward_est.copy()
        kf = KalmanFilter1D(process_noise, meas_noise, dt)
        last_valid = np.where(mask)[0][-1]
        kf.init(points[last_valid, axis], 0.0)

        for i in reversed(range(len(points))):
            kf.predict()
            if mask[i]:
                backward_est[i] = kf.update(points[i, axis])
            else:
                backward_est[i] = kf.x[0]

        # Усредняем forward и backward
        smoothed[:, axis] = (forward_est + backward_est) / 2

    return smoothed


def main() -> None:
    if not INPUT_RAW.exists():
        raise FileNotFoundError(
            f"Сырая траектория не найдена: {INPUT_RAW}. Сначала запустите triangulate.py"
        )

    print("Загрузка сырой траектории...")
    frames, points, mask = load_trajectory(INPUT_RAW)
    n_total = len(frames)
    n_tri = mask.sum()
    print(f"  Всего кадров: {n_total}, триангулировано: {n_tri}, пропусков: {n_total - n_tri}")

    log_rows = []

    # 1. Удаление выбросов
    print("1. Удаление выбросов...")
    points_clean = remove_outliers(frames, points, mask.copy())
    n_removed = (~mask & (points_clean[:, 0] != points_clean[:, 0])).sum()  # новые NaN
    # Правильно считаем: сколько стало False в mask
    mask_before = mask.copy()
    mask = ~np.isnan(points_clean[:, 0])
    n_outliers = (mask_before & ~mask).sum()
    print(f"   Удалено выбросов: {n_outliers}")
    for i in range(n_total):
        log_rows.append({"frame": frames[i], "stage": "outlier_removal",
                         "removed": not mask[i] and mask_before[i]})

    # 2. Интерполяция пропусков
    print("2. Интерполяция пропусков...")
    points_interp = interpolate_gaps(frames, points_clean, mask)
    n_filled = (~mask).sum()
    mask = np.ones(n_total, dtype=bool)  # после интерполяции всё заполнено
    print(f"   Заполнено пропусков: {n_filled}")

    # 3. Сглаживание Савицкого-Голея
    print("3. Сглаживание (Savitzky-Golay)...")
    points_sg = savgol_smooth(frames, points_interp, mask)
    print(f"   Окно: {SG_WINDOW}, порядок: {SG_POLYORDER}")

    # 4. Фильтр Калмана (опционально, комментируйте если не нужно)
    print("4. Фильтр Калмана (forward-backward)...")
    points_kalman = kalman_smooth(frames, points_sg, np.ones(n_total, dtype=bool))
    final_points = points_kalman

    # Сохраняем итоговую траекторию
    with open(OUTPUT_SMOOTH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z"])
        for frame, pt in zip(frames, final_points):
            writer.writerow([frame, f"{pt[0]:.3f}", f"{pt[1]:.3f}", f"{pt[2]:.3f}"])

    # Лог постобработки
    with open(OUTPUT_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "stage", "removed"])
        for r in log_rows:
            writer.writerow([r["frame"], r["stage"], r["removed"]])

    print(f"\nГотово: сглаженная траектория -> {OUTPUT_SMOOTH}")
    print(f"Лог постобработки -> {OUTPUT_LOG}")

    # Статистика
    print(f"\nСтатистика итоговой траектории:")
    print(f"  X: [{final_points[:,0].min():.1f}, {final_points[:,0].max():.1f}]")
    print(f"  Y: [{final_points[:,1].min():.1f}, {final_points[:,1].max():.1f}]")
    print(f"  Z: [{final_points[:,2].min():.1f}, {final_points[:,2].max():.1f}]")


if __name__ == "__main__":
    main()