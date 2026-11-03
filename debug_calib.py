"""
debug_calib.py
============================================================
Диагностика калибровки: проецирует GT 3D-точки на кадры видео
и сохраняет визуализацию для проверки конвенции осей.

Запуск:
    python debug_calib.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from camera_calib import (
    VIDEO_WIDTH, VIDEO_HEIGHT,
    load_transform_csv, _num, build_intrinsics, build_camera_poses,
    project_point, Intrinsics, CameraPose,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "runs" / "calib_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_FILES = {
    1: ("object_transform_CineCameraActor1.csv", "Way1.FinalImage.CineCameraActor1.mp4"),
    2: ("object_transform_CineCameraActor2.csv", "Way1.FinalImage.CineCameraActor2.mp4"),
    3: ("object_transform_CineCameraActor3.csv", "Way1.FinalImage.CineCameraActor3.mp4"),
    4: ("object_transform_CineCameraActor4.csv", "Way1.FinalImage.CineCameraActor4.mp4"),
    5: ("object_transform_CineCameraActor5.csv", "Way1.FinalImage.CineCameraActor5.mp4"),
}

GT_FILE = "object_transform_SM_AA_track.csv"

# Кадры для проверки (начало, середина, конец)
TEST_FRAMES = [0, 10, 20, 30, 50, 100, 200, 500, 800, 900, 1000]


def load_gt_points() -> dict[int, np.ndarray]:
    """Читает GT траекторию."""
    rows, _ = load_transform_csv(str(DATA_DIR / GT_FILE))
    points = {}
    for row in rows:
        try:
            frame = int(row[0])
            x = _num(row[1]); y = _num(row[2]); z = _num(row[3])
            if x is not None and y is not None and z is not None:
                points[frame] = np.array([x, y, z], dtype=float)
        except (ValueError, IndexError):
            continue
    return points


def load_camera(cam_id: int) -> tuple[Intrinsics, dict[int, CameraPose], float]:
    """Загружает интринсики и позы камеры."""
    csv_file, _ = CAMERA_FILES[cam_id]
    rows, header = load_transform_csv(str(DATA_DIR / csv_file))

    # Извлекаем фокус из заголовка
    focal = None
    for cell in reversed(header):
        cell = cell.strip()
        if cell and cell.replace('.', '').replace(',', '').isdigit():
            focal = float(cell.replace(',', '.'))
            break
    if focal is None:
        raise ValueError(f"Не найден фокус в заголовке: {header}")

    intrinsics = build_intrinsics(focal)
    poses = {p.frame: p for p in build_camera_poses(rows, intrinsics)}
    return intrinsics, poses, focal


def project_gt_to_camera(
    gt_points: dict[int, np.ndarray],
    intrinsics: Intrinsics,
    poses: dict[int, CameraPose],
    frames: list[int],
) -> dict[int, tuple[float, float] | None]:
    """Проецирует GT точки на камеру для заданных кадров."""
    projections = {}
    for frame in frames:
        if frame not in gt_points:
            projections[frame] = None
            continue
        if frame not in poses:
            projections[frame] = None
            continue
        pt = gt_points[frame]
        pose = poses[frame]
        uv = project_point(intrinsics, pose, pt)
        projections[frame] = uv
    return projections


def visualize_on_video(
    cam_id: int,
    projections: dict[int, tuple[float, float] | None],
    gt_points: dict[int, np.ndarray],
) -> None:
    """Рисует проекции GT на видео и сохраняет кадры."""
    _, video_file = CAMERA_FILES[cam_id]
    video_path = DATA_DIR / video_file

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Не удалось открыть {video_path}")
        return

    for frame_idx, uv in projections.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Кадр {frame_idx}: не прочитан")
            continue

        # Рисуем проекцию GT (зелёный круг)
        if uv is not None:
            u, v = uv
            if 0 <= u < VIDEO_WIDTH and 0 <= v < VIDEO_HEIGHT:
                cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 0), 3)
                cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
                # Подпись с 3D координатами
                pt = gt_points[frame_idx]
                cv2.putText(frame, f"GT: ({pt[0]:.0f},{pt[1]:.0f},{pt[2]:.0f})",
                            (int(u)+15, int(v)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, f"GT за кадром: ({u:.0f},{v:.0f})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Нет позы камеры или GT", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Номер кадра
        cv2.putText(frame, f"Cam{cam_id} Frame: {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Сохраняем
        out_path = OUT_DIR / f"cam{cam_id}_frame{frame_idx:04d}_debug.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()
    print(f"  Камера {cam_id}: сохранено {len(projections)} отладочных кадров в {OUT_DIR}")


def check_camera_pose(cam_id: int, poses: dict[int, CameraPose], frame: int = 0) -> None:
    """Выводит детали позы камеры для проверки."""
    if frame not in poses:
        print(f"  Камера {cam_id}: нет позы для кадра {frame}")
        return
    pose = poses[frame]
    print(f"\n=== Камера {cam_id}, кадр {frame} ===")
    print(f"  Loc (cm): {pose.loc.flatten()}")
    print(f"  Rot (deg): {pose.rot}")
    print(f"  R matrix:\n{pose.R}")
    print(f"  W2C matrix:\n{pose.W2C}")
    print(f"  Forward (col0): {pose.R[:, 0]}")
    print(f"  Right (col1):   {pose.R[:, 1]}")
    print(f"  Up (col2):      {pose.R[:, 2]}")
    print(f"  W2C rows (right, down, forward):")
    print(f"    right:  {pose.W2C[0]}")
    print(f"    down:   {pose.W2C[1]}")
    print(f"    forward:{pose.W2C[2]}")


def main():
    print("=" * 60)
    print("ДИАГНОСТИКА КАЛИБРОВКИ: проекция GT на видео")
    print("=" * 60)

    # Загружаем GT
    gt_points = load_gt_points()
    print(f"\nGT точек загружено: {len(gt_points)}")
    if gt_points:
        frames_sorted = sorted(gt_points.keys())
        print(f"  Диапазон кадров: {frames_sorted[0]} – {frames_sorted[-1]}")
        print(f"  Первая точка: {gt_points[frames_sorted[0]]}")
        print(f"  Последняя точка: {gt_points[frames_sorted[-1]]}")

    # Проверяем каждую камеру
    for cam_id in range(1, 6):
        print(f"\n{'='*40}")
        print(f"КАМЕРА {cam_id}")
        print(f"{'='*40}")

        try:
            intrinsics, poses, focal = load_camera(cam_id)
            print(f"  Фокус: {focal} мм")
            print(f"  Интринсики K:\n{intrinsics.K}")
            print(f"  Поз загружено: {len(poses)}")

            # Детали позы кадра 0
            check_camera_pose(cam_id, poses, 0)

            # Проецируем GT на тестовые кадры
            test_frames = [f for f in TEST_FRAMES if f in gt_points]
            projections = project_gt_to_camera(gt_points, intrinsics, poses, test_frames)

            print(f"\n  Проекции GT:")
            for frame in test_frames:
                uv = projections.get(frame)
                pt = gt_points.get(frame)
                if uv:
                    u, v = uv
                    in_frame = 0 <= u < VIDEO_WIDTH and 0 <= v < VIDEO_HEIGHT
                    status = "В КАДРЕ" if in_frame else "ЗА КАДРОМ"
                    print(f"    Кадр {frame:4d}: 3D=({pt[0]:8.0f},{pt[1]:8.0f},{pt[2]:6.0f}) -> 2D=({u:7.1f},{v:7.1f}) [{status}]")
                else:
                    print(f"    Кадр {frame:4d}: нет данных камеры или GT")

            # Визуализация на видео
            visualize_on_video(cam_id, projections, gt_points)

        except Exception as e:
            print(f"  ОШИБКА: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Готово! Отладочные кадры в: {OUT_DIR}")
    print(f"Откройте их и проверьте: попадают ли зелёные круги в грузовик?")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()