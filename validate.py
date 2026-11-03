"""
validate.py
============================================================
Этап 6: Валидация восстановленной траектории против Ground Truth.

Сравнивает trajectory_smooth.csv (или trajectory_raw.csv) с
object_transform_SM_AA_track.csv (GT).

Метрики:
  - MAE / RMSE / Max ошибка по 3D (метры)
  - Ошибки по осям X, Y, Z (метры)
  - Репроекционная ошибка: спроецировать восстановленную 3D обратно
    в камеры и сравнить с измеренными 2D-детекциями (пиксели)
  - Процент кадров с успешной триангуляцией

Выход:
  - runs/validation/validation_report.csv (по-кадровая ошибка)
  - runs/validation/validation_summary.json (агрегаты)
  - runs/validation/reprojection_report.csv (репроекционная ошибка)
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
)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"
VALIDATION_DIR = RUNS_DIR / "validation"

# Входные файлы
GT_FILE = DATA_DIR / "object_transform_SM_AA_track.csv"
TRAJ_SMOOTH = TRAJECTORY_DIR / "trajectory_smooth.csv"
TRAJ_RAW = TRAJECTORY_DIR / "trajectory_raw.csv"
DETECTIONS_FILE = RUNS_DIR / "detections" / "all_detections.csv"

# Выходные файлы
REPORT_CSV = VALIDATION_DIR / "validation_report.csv"
SUMMARY_JSON = VALIDATION_DIR / "validation_summary.json"
REPROJ_CSV = VALIDATION_DIR / "reprojection_report.csv"

CAMERA_FILES = {
    1: "object_transform_CineCameraActor1.csv",
    2: "object_transform_CineCameraActor2.csv",
    3: "object_transform_CineCameraActor3.csv",
    4: "object_transform_CineCameraActor4.csv",
    5: "object_transform_CineCameraActor5.csv",
}


def load_ground_truth(path: Path) -> dict[int, np.ndarray]:
    """Читает GT траекторию: frame -> (x, y, z) в см (UE)."""
    gt = {}
    rows, _ = load_transform_csv(str(path))
    for row in rows:
        try:
            frame = int(row[0])
            x = _num(row[1])
            y = _num(row[2])
            z = _num(row[3])
            if x is not None and y is not None and z is not None:
                gt[frame] = np.array([x, y, z], dtype=float)
        except (ValueError, IndexError):
            continue
    return gt


def load_trajectory(path: Path) -> dict[int, np.ndarray]:
    """Читает восстановленную траекторию: frame -> (x, y, z)."""
    traj = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            if row["x"] == "" or row["x"] is None:
                continue
            traj[frame] = np.array([
                float(row["x"]), float(row["y"]), float(row["z"])
            ], dtype=float)
    return traj


def load_detections(path: Path) -> dict[int, list[dict]]:
    """Читает детекции для репроекционной ошибки."""
    dets: dict[int, list[dict]] = {}
    if not path.exists():
        return dets
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            dets.setdefault(frame, []).append({
                "camera_id": int(row["camera_id"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
            })
    return dets


def load_all_cameras() -> dict[int, tuple[Intrinsics, dict[int, CameraPose]]]:
    """Загружает камеры для репроекции."""
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
    return cameras


def compute_reprojection_error(
    frame: int,
    X_est: np.ndarray,
    cameras: dict[int, tuple[Intrinsics, dict[int, CameraPose]]],
    detections: dict[int, list[dict]],
) -> list[dict]:
    """Вычисляет репроекционную ошибку для оцененной 3D-точки."""
    errors = []
    if frame not in detections:
        return errors

    for det in detections[frame]:
        cam_id = det["camera_id"]
        if cam_id not in cameras:
            continue
        intrinsics, poses = cameras[cam_id]
        pose = poses.get(frame)
        if pose is None:
            continue

        uv = project_point(intrinsics, pose, X_est)
        if uv is None:
            continue

        u_est, v_est = uv
        u_gt, v_gt = det["cx"], det["cy"]
        err_px = np.sqrt((u_est - u_gt)**2 + (v_est - v_gt)**2)

        errors.append({
            "frame": frame,
            "camera_id": cam_id,
            "u_est": u_est,
            "v_est": v_est,
            "u_gt": u_gt,
            "v_gt": v_gt,
            "error_px": err_px,
        })
    return errors


def main(trajectory_file: Optional[Path] = None) -> None:
    # Выбираем траекторию: сглаженная приоритетнее, потом сырая
    if trajectory_file is None:
        if TRAJ_SMOOTH.exists():
            trajectory_file = TRAJ_SMOOTH
        elif TRAJ_RAW.exists():
            trajectory_file = TRAJ_RAW
        else:
            raise FileNotFoundError(
                f"Траектория не найдена. Проверьте {TRAJ_SMOOTH} или {TRAJ_RAW}"
            )

    print(f"Валидация траектории: {trajectory_file}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    print("Загрузка Ground Truth...")
    gt = load_ground_truth(GT_FILE)
    print(f"  GT кадров: {len(gt)}")

    print("Загрузка восстановленной траектории...")
    traj = load_trajectory(trajectory_file)
    print(f"  Восстановлено кадров: {len(traj)}")

    print("Загрузка детекций (для репроекции)...")
    detections = load_detections(DETECTIONS_FILE)
    print(f"  Кадров с детекциями: {len(detections)}")

    print("Загрузка камер...")
    cameras = load_all_cameras()

    # Общие кадры
    common_frames = sorted(set(gt.keys()) & set(traj.keys()))
    gt_only = sorted(set(gt.keys()) - set(traj.keys()))
    traj_only = sorted(set(traj.keys()) - set(gt.keys()))

    print(f"\nОбщих кадров: {len(common_frames)}")
    print(f"Только в GT: {len(gt_only)}")
    print(f"Только в восстановленной: {len(traj_only)}")

    if not common_frames:
        print("Нет общих кадров для сравнения!")
        return

    # По-кадровая 3D ошибка
    errors_3d = []
    reproj_errors = []

    for frame in common_frames:
        X_gt = gt[frame]
        X_est = traj[frame]

        dx = X_est[0] - X_gt[0]
        dy = X_est[1] - X_gt[1]
        dz = X_est[2] - X_gt[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)  # в см

        errors_3d.append({
            "frame": frame,
            "gt_x": X_gt[0], "gt_y": X_gt[1], "gt_z": X_gt[2],
            "est_x": X_est[0], "est_y": X_est[1], "est_z": X_est[2],
            "dx_cm": dx, "dy_cm": dy, "dz_cm": dz,
            "error_cm": dist,
            "error_m": dist / 100.0,
        })

        # Репроекционная ошибка
        reproj_errors.extend(compute_reprojection_error(frame, X_est, cameras, detections))

    # Сохраняем отчёт 3D ошибки
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "gt_x", "gt_y", "gt_z",
            "est_x", "est_y", "est_z",
            "dx_cm", "dy_cm", "dz_cm", "error_cm", "error_m"
        ])
        for e in errors_3d:
            writer.writerow([
                e["frame"], e["gt_x"], e["gt_y"], e["gt_z"],
                e["est_x"], e["est_y"], e["est_z"],
                e["dx_cm"], e["dy_cm"], e["dz_cm"],
                e["error_cm"], e["error_m"]
            ])

    # Сохраняем репроекционную ошибку
    with open(REPROJ_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_id", "u_est", "v_est", "u_gt", "v_gt", "error_px"])
        for e in reproj_errors:
            writer.writerow([e["frame"], e["camera_id"], e["u_est"], e["v_est"],
                            e["u_gt"], e["v_gt"], e["error_px"]])

    # Агрегированные метрики
    errs_cm = [e["error_cm"] for e in errors_3d]
    errs_m = [e["error_m"] for e in errors_3d]
    dxs = [abs(e["dx_cm"]) for e in errors_3d]
    dys = [abs(e["dy_cm"]) for e in errors_3d]
    dzs = [abs(e["dz_cm"]) for e in errors_3d]

    mean_err_cm = np.mean(errs_cm)
    rmse_cm = np.sqrt(np.mean(np.array(errs_cm)**2))
    max_err_cm = np.max(errs_cm)
    median_err_cm = np.median(errs_cm)
    std_err_cm = np.std(errs_cm)

    # Репроекционная ошибка
    if reproj_errors:
        reproj_px = [e["error_px"] for e in reproj_errors]
        mean_reproj = np.mean(reproj_px)
        rmse_reproj = np.sqrt(np.mean(np.array(reproj_px)**2))
        max_reproj = np.max(reproj_px)
    else:
        mean_reproj = rmse_reproj = max_reproj = 0.0

    print(f"\n=== VALIDATION REPORT ===")
    print(f"Траектория: {trajectory_file.name}")
    print(f"Общих кадров: {len(common_frames)}")
    print(f"\n--- 3D ошибка (в см) ---")
    print(f"  Mean (MAE):  {mean_err_cm:.2f} см")
    print(f"  Median:      {median_err_cm:.2f} см")
    print(f"  RMSE:        {rmse_cm:.2f} см")
    print(f"  Std:         {std_err_cm:.2f} см")
    print(f"  Max:         {max_err_cm:.2f} см")
    print(f"\n--- По осям (MAE, см) ---")
    print(f"  |dX|: {np.mean(dxs):.2f}")
    print(f"  |dY|: {np.mean(dys):.2f}")
    print(f"  |dZ|: {np.mean(dzs):.2f}")
    print(f"\n--- Репроекционная ошибка (пиксели) ---")
    print(f"  Mean:  {mean_reproj:.2f} px")
    print(f"  RMSE:  {rmse_reproj:.2f} px")
    print(f"  Max:   {max_reproj:.2f} px")
    print(f"  N:     {len(reproj_errors)} измерений")

    # Сохраняем summary JSON
    summary = {
        "trajectory_file": str(trajectory_file),
        "common_frames": len(common_frames),
        "gt_only_frames": len(gt_only),
        "traj_only_frames": len(traj_only),
        "error_3d_cm": {
            "mean": float(mean_err_cm),
            "median": float(median_err_cm),
            "rmse": float(rmse_cm),
            "std": float(std_err_cm),
            "max": float(max_err_cm),
            "mean_abs_dx": float(np.mean(dxs)),
            "mean_abs_dy": float(np.mean(dys)),
            "mean_abs_dz": float(np.mean(dzs)),
        },
        "reprojection_error_px": {
            "mean": float(mean_reproj),
            "rmse": float(rmse_reproj),
            "max": float(max_reproj),
            "n_measurements": len(reproj_errors),
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nФайлы сохранены:")
    print(f"  {REPORT_CSV}")
    print(f"  {REPROJ_CSV}")
    print(f"  {SUMMARY_JSON}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate trajectory against GT")
    parser.add_argument("--traj", type=str, default=None,
                        help="Path to trajectory CSV (default: smooth > raw)")
    args = parser.parse_args()

    traj_path = Path(args.traj) if args.traj else None
    main(trajectory_file=traj_path)