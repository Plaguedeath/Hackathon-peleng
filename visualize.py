"""
visualize.py
============================================================
Этап 7: Визуализация результатов.

1. Видео с наложенными 2D-детекциями и репроекциями 3D-точек
2. 3D-анимация траектории объекта + позиции 5 камер (MP4)
3. Графики ошибок (3D ошибка по кадрам, репроекционная ошибка)

Выход:
  - runs/visualization/cam{N}_overlay.mp4  (видео с боксами + репроекция)
  - runs/visualization/trajectory_3d.mp4   (3D анимация MP4)
  - runs/visualization/error_plots.png     (графики ошибок)
  - runs/visualization/summary_metrics.png (сводный барплот метрик)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"
VALIDATION_DIR = RUNS_DIR / "validation"
DETECTIONS_DIR = RUNS_DIR / "detections"
VIS_DIR = RUNS_DIR / "visualization"

TRAJ_SMOOTH = TRAJECTORY_DIR / "trajectory_smooth.csv"
TRAJ_RAW = TRAJECTORY_DIR / "trajectory_raw.csv"
DETECTIONS_FILE = DETECTIONS_DIR / "all_detections.csv"
VALIDATION_REPORT = VALIDATION_DIR / "validation_report.csv"
REPROJ_REPORT = VALIDATION_DIR / "reprojection_report.csv"
SUMMARY_JSON = VALIDATION_DIR / "validation_summary.json"

CAMERA_FILES = {
    1: "object_transform_CineCameraActor1.csv",
    2: "object_transform_CineCameraActor2.csv",
    3: "object_transform_CineCameraActor3.csv",
    4: "object_transform_CineCameraActor4.csv",
    5: "object_transform_CineCameraActor5.csv",
}

VIDEO_FILES = {
    1: DATA_DIR / "Way1.FinalImage.CineCameraActor1.mp4",
    2: DATA_DIR / "Way1.FinalImage.CineCameraActor2.mp4",
    3: DATA_DIR / "Way1.FinalImage.CineCameraActor3.mp4",
    4: DATA_DIR / "Way1.FinalImage.CineCameraActor4.mp4",
    5: DATA_DIR / "Way1.FinalImage.CineCameraActor5.mp4",
}

GT_FILE = DATA_DIR / "object_transform_SM_AA_track.csv"


def load_trajectory(path: Path) -> dict[int, np.ndarray]:
    """Читает траекторию: frame -> (x, y, z)."""
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


def load_gt(path: Path) -> dict[int, np.ndarray]:
    """Читает GT траекторию."""
    from camera_calib import load_transform_csv, _num
    gt = {}
    rows, _ = load_transform_csv(str(path))
    for row in rows:
        try:
            frame = int(row[0])
            x = _num(row[1]); y = _num(row[2]); z = _num(row[3])
            if x is not None and y is not None and z is not None:
                gt[frame] = np.array([x, y, z], dtype=float)
        except (ValueError, IndexError):
            continue
    return gt


def load_camera_positions() -> dict[int, np.ndarray]:
    """Возвращает позиции камер (x, y, z) в см."""
    from camera_calib import load_transform_csv, _num, build_intrinsics, build_camera_poses
    positions = {}
    for cam_id, fname in CAMERA_FILES.items():
        rows, header = load_transform_csv(str(DATA_DIR / fname))
        focal = None
        for cell in reversed(header):
            cell = cell.strip()
            if cell and cell.replace('.', '').isdigit():
                focal = float(cell)
                break
        if focal is None:
            continue
        intrinsics = build_intrinsics(focal)
        poses = {p.frame: p for p in build_camera_poses(rows, intrinsics)}
        # Берём позицию из первого кадра (статичны, кроме Cam5)
        first_pose = poses[min(poses.keys())]
        positions[cam_id] = first_pose.loc.flatten()
    return positions


def load_detections(path: Path) -> dict[int, list[dict]]:
    """Читает детекции для отрисовки боксов."""
    dets: dict[int, list[dict]] = {}
    if not path.exists():
        return dets
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            dets.setdefault(frame, []).append({
                "camera_id": int(row["camera_id"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "confidence": float(row["confidence"]),
            })
    return dets


def load_reprojections(path: Path) -> dict[int, list[dict]]:
    """Читает репроекционные ошибки для отрисовки."""
    reps: dict[int, list[dict]] = {}
    if not path.exists():
        return reps
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            reps.setdefault(frame, []).append({
                "camera_id": int(row["camera_id"]),
                "u_est": float(row["u_est"]),
                "v_est": float(row["v_est"]),
                "u_gt": float(row["u_gt"]),
                "v_gt": float(row["v_gt"]),
                "error_px": float(row["error_px"]),
            })
    return reps


def create_overlay_videos(
    detections: dict[int, list[dict]],
    reprojections: dict[int, list[dict]],
    max_frames: int | None = None,
) -> None:
    """Создаёт видео с наложенными боксами детекций и репроекциями 3D."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for cam_id, video_path in VIDEO_FILES.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Не удалось открыть {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total = min(total, max_frames)

        out_path = VIS_DIR / f"cam{cam_id}_overlay.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        print(f"  Создание {out_path.name} ({total} кадров)...")

        idx = 0
        while idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            # Рисуем детекции YOLO (зелёные боксы)
            if idx in detections:
                for det in detections[idx]:
                    if det["camera_id"] != cam_id:
                        continue
                    x1, y1 = int(det["x1"]), int(det["y1"])
                    x2, y2 = int(det["x2"]), int(det["y2"])
                    conf = det["confidence"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"YOLO {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Рисуем репроекции 3D (красные кресты)
            if idx in reprojections:
                for rep in reprojections[idx]:
                    if rep["camera_id"] != cam_id:
                        continue
                    u, v = int(rep["u_est"]), int(rep["v_est"])
                    err = rep["error_px"]
                    # Крест репроекции
                    cv2.drawMarker(frame, (u, v), (0, 0, 255),
                                   cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(frame, f"3D->2D err={err:.1f}px", (u + 10, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # GT точка (синий круг) — если есть
                    u_gt, v_gt = int(rep["u_gt"]), int(rep["v_gt"])
                    cv2.circle(frame, (u_gt, v_gt), 5, (255, 0, 0), 2)

            # Номер кадра
            cv2.putText(frame, f"Frame: {idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            writer.write(frame)
            idx += 1

        cap.release()
        writer.release()
        print(f"    Готово: {out_path}")


def create_3d_animation(
    traj_smooth: dict[int, np.ndarray],
    gt: dict[int, np.ndarray],
    cam_positions: dict[int, np.ndarray],
    output_path: Path,
    fps: int = 15,
    max_frames: int = 300,
) -> None:
    """Создаёт 3D MP4 анимацию траектории + камер (MP4 вместо GIF для экономии памяти)."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    frames_sorted = sorted(set(traj_smooth.keys()) | set(gt.keys()))
    if not frames_sorted:
        print("  Нет данных для анимации")
        return

    # Ограничиваем число кадров для памяти
    if len(frames_sorted) > max_frames:
        step = len(frames_sorted) // max_frames
        frames_sorted = frames_sorted[::step]
        print(f"  Ограничение кадров анимации: {len(frames_sorted)} (шаг {step})")

    # Определяем границы
    all_points = list(traj_smooth.values()) + list(gt.values()) + list(cam_positions.values())
    all_points = np.array(all_points)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # Добавляем отступы
    pad = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad
    z_min -= pad; z_max += pad

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Трассы
    traj_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Reconstructed (smooth)')
    gt_line, = ax.plot([], [], [], 'g-', linewidth=1, alpha=0.7, label='Ground Truth')
    traj_point, = ax.plot([], [], [], 'bo', markersize=8)
    gt_point, = ax.plot([], [], [], 'go', markersize=6)

    # Камеры
    cam_x = [pos[0] for pos in cam_positions.values()]
    cam_y = [pos[1] for pos in cam_positions.values()]
    cam_z = [pos[2] for pos in cam_positions.values()]
    ax.scatter(cam_x, cam_y, cam_z, c='red', s=100, marker='^', label='Cameras')

    # Подписи камер
    for cam_id, pos in cam_positions.items():
        ax.text(pos[0], pos[1], pos[2], f' Cam{cam_id}', color='red', fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('3D Trajectory Reconstruction')
    ax.legend()

    def update(frame_idx):
        frame = frames_sorted[frame_idx]

        # Накопленные траектории до текущего кадра
        past_frames = [f for f in frames_sorted if f <= frame]
        if past_frames:
            traj_pts = np.array([traj_smooth[f] for f in past_frames if f in traj_smooth])
            gt_pts = np.array([gt[f] for f in past_frames if f in gt])

            if len(traj_pts) > 0:
                traj_line.set_data(traj_pts[:, 0], traj_pts[:, 1])
                traj_line.set_3d_properties(traj_pts[:, 2])
                traj_point.set_data([traj_pts[-1, 0]], [traj_pts[-1, 1]])
                traj_point.set_3d_properties([traj_pts[-1, 2]])

            if len(gt_pts) > 0:
                gt_line.set_data(gt_pts[:, 0], gt_pts[:, 1])
                gt_line.set_3d_properties(gt_pts[:, 2])
                gt_point.set_data([gt_pts[-1, 0]], [gt_pts[-1, 1]])
                gt_point.set_3d_properties([gt_pts[-1, 2]])

        ax.set_title(f'3D Trajectory — Frame {frame}')
        return traj_line, gt_line, traj_point, gt_point

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(frames_sorted),
                         interval=1000/fps, blit=False, repeat=True)

    # Сохраняем MP4 (ffmpeg writer) вместо GIF — экономит память
    # Сначала пробуем ffmpeg, если нет — PillowWriter с ограничением кадров
    try:
        anim.save(str(output_path.with_suffix('.mp4')), writer='ffmpeg', fps=fps, dpi=80,
                   extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  3D анимация сохранена (MP4): {output_path.with_suffix('.mp4')}")
    except Exception as e:
        print(f"  FFmpeg недоступен ({e}), пробуем GIF с пониженным качеством...")
        # Фоллбэк: GIF с пониженным dpi и меньше кадров
        try:
            anim.save(str(output_path), writer='pillow', fps=min(fps, 10), dpi=60)
            print(f"  3D анимация сохранена (GIF): {output_path}")
        except MemoryError:
            print(f"  ⚠ Недостаточно памяти для анимации, пропускаем")
    finally:
        plt.close(fig)


def create_error_plots(
    validation_report: Path,
    reprojection_report: Path,
    output_path: Path,
) -> None:
    """Строит графики ошибок."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем данные
    frames_3d, errors_cm, errors_m = [], [], []
    dx_cm, dy_cm, dz_cm = [], [], []

    if validation_report.exists():
        with open(validation_report, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames_3d.append(int(row["frame"]))
                errors_cm.append(float(row["error_cm"]))
                errors_m.append(float(row["error_m"]))
                dx_cm.append(abs(float(row["dx_cm"])))
                dy_cm.append(abs(float(row["dy_cm"])))
                dz_cm.append(abs(float(row["dz_cm"])))

    frames_rep, errors_px = [], []
    if reprojection_report.exists():
        with open(reprojection_report, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames_rep.append(int(row["frame"]))
                errors_px.append(float(row["error_px"]))

    # Создаём графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validation Metrics', fontsize=14)

    # 1. 3D ошибка по кадрам
    ax = axes[0, 0]
    if frames_3d:
        ax.plot(frames_3d, errors_cm, 'b-', linewidth=0.8, label='Error (cm)')
        ax.axhline(np.mean(errors_cm), color='r', linestyle='--', label=f'Mean: {np.mean(errors_cm):.1f} cm')
        ax.set_xlabel('Frame')
        ax.set_ylabel('3D Error (cm)')
        ax.set_title('3D Trajectory Error per Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Ошибки по осям
    ax = axes[0, 1]
    if frames_3d:
        ax.plot(frames_3d, dx_cm, 'r-', linewidth=0.6, label='|dX|', alpha=0.8)
        ax.plot(frames_3d, dy_cm, 'g-', linewidth=0.6, label='|dY|', alpha=0.8)
        ax.plot(frames_3d, dz_cm, 'b-', linewidth=0.6, label='|dZ|', alpha=0.8)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Absolute Error (cm)')
        ax.set_title('Per-Axis Absolute Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Гистограмма 3D ошибки
    ax = axes[1, 0]
    if errors_cm:
        ax.hist(errors_cm, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors_cm), color='r', linestyle='--', label=f'Mean: {np.mean(errors_cm):.1f} cm')
        ax.axvline(np.median(errors_cm), color='g', linestyle='--', label=f'Median: {np.median(errors_cm):.1f} cm')
        ax.set_xlabel('3D Error (cm)')
        ax.set_ylabel('Frequency')
        ax.set_title('3D Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Репроекционная ошибка
    ax = axes[1, 1]
    if errors_px:
        ax.plot(frames_rep, errors_px, 'm-', linewidth=0.6, alpha=0.7)
        ax.axhline(np.mean(errors_px), color='r', linestyle='--', label=f'Mean: {np.mean(errors_px):.1f} px')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Reprojection Error (px)')
        ax.set_title('Reprojection Error per Measurement')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Графики ошибок сохранены: {output_path}")


def create_summary_plot(summary_json: Path, output_path: Path) -> None:
    """Создаёт сводный график с ключевыми метриками."""
    if not summary_json.exists():
        return

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_3d = summary.get("error_3d_cm", {})
    metrics_rep = summary.get("reprojection_error_px", {})

    labels = ['MAE (cm)', 'Median (cm)', 'RMSE (cm)', 'Max (cm)',
              '|dX| (cm)', '|dY| (cm)', '|dZ| (cm)',
              'Reproj MAE (px)', 'Reproj RMSE (px)', 'Reproj Max (px)']
    values = [
        metrics_3d.get('mean', 0), metrics_3d.get('median', 0),
        metrics_3d.get('rmse', 0), metrics_3d.get('max', 0),
        metrics_3d.get('mean_abs_dx', 0), metrics_3d.get('mean_abs_dy', 0),
        metrics_3d.get('mean_abs_dz', 0),
        metrics_rep.get('mean', 0), metrics_rep.get('rmse', 0),
        metrics_rep.get('max', 0),
    ]
    colors = ['blue'] * 7 + ['magenta'] * 3

    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value')
    ax.set_title('Validation Summary Metrics')
    ax.grid(True, axis='x', alpha=0.3)

    # Подписи значений на барах
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01 * max(values), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Сводный график сохранен: {output_path}")


def main(
    max_video_frames: int | None = None,
    create_videos: bool = True,
    create_3d: bool = True,
    create_plots: bool = True,
) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Выбираем траекторию
    if TRAJ_SMOOTH.exists():
        traj_path = TRAJ_SMOOTH
    elif TRAJ_RAW.exists():
        traj_path = TRAJ_RAW
    else:
        print("Траектория не найдена. Запустите triangulate.py / postprocess.py")
        return

    print("Загрузка данных для визуализации...")
    traj_smooth = load_trajectory(traj_path)
    gt = load_gt(GT_FILE)
    cam_positions = load_camera_positions()
    detections = load_detections(DETECTIONS_FILE)
    reprojections = load_reprojections(REPROJ_REPORT)

    print(f"  Траектория: {len(traj_smooth)} кадров")
    print(f"  GT: {len(gt)} кадров")
    print(f"  Камеры: {len(cam_positions)}")
    print(f"  Детекции: {sum(len(v) for v in detections.values())} шт.")
    print(f"  Репроекции: {sum(len(v) for v in reprojections.values())} шт.")

    # 1. Видео с наложениями
    if create_videos:
        print("\nСоздание видео с наложениями...")
        create_overlay_videos(detections, reprojections, max_frames=max_video_frames)

    # 2. 3D анимация
    if create_3d:
        print("\nСоздание 3D анимации...")
        anim_path = VIS_DIR / "trajectory_3d.gif"
        create_3d_animation(traj_smooth, gt, cam_positions, anim_path)

    # 3. Графики ошибок
    if create_plots:
        print("\nСоздание графиков ошибок...")
        create_error_plots(VALIDATION_REPORT, REPROJ_REPORT, VIS_DIR / "error_plots.png")
        create_summary_plot(SUMMARY_JSON, VIS_DIR / "summary_metrics.png")

    print(f"\nВсе артефакты визуализации в: {VIS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize pipeline results")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for video overlay")
    parser.add_argument("--no-videos", action="store_true", help="Skip video overlay creation")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D animation")
    parser.add_argument("--no-plots", action="store_true", help="Skip error plots")
    args = parser.parse_args()

    main(
        max_video_frames=args.max_frames,
        create_videos=not args.no_videos,
        create_3d=not args.no_3d,
        create_plots=not args.no_plots,
    )