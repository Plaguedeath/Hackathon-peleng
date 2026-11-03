"""
Visualizer — визуализация результатов: overlay на видео, 3D траектории, графики.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from camera_manager import CameraManager, Camera
from detector import Detection2D, CameraDetections
from tracker3d import Track3D
from triangulator import TriangulationResult


# =============================================================================
# 2D Overlay Visualizer
# =============================================================================

@dataclass
class Visualizer2D:
    """Рисование 2D наложений на кадрах видео."""

    camera_manager: CameraManager
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (0, 128, 128), (128, 128, 0)
    ])
    draw_boxes: bool = True
    draw_reprojection: bool = True
    draw_ids: bool = True
    draw_trails: bool = True
    max_trail_length: int = 50
    box_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 2

    # История репроекций для траилов
    _reprojection_history: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 128), (0, 128, 128), (128, 128, 0)
            ]

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]

    def draw_detections(
        self,
        frame: np.ndarray,
        cam_id: int,
        detections: CameraDetections,
        tracks_3d: Optional[Dict[int, Track3D]] = None
    ) -> np.ndarray:
        """Рисует 2D детекции и репроекции 3D треков на кадре."""
        vis = frame.copy()

        # 1. Рисуем YOLO боксы
        if self.draw_boxes:
            for det in detections.detections:
                color = self._get_color(det.track_id if det.track_id >= 0 else det.class_id)
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, self.box_thickness)

                # Лейбл: ID + класс + confidence
                if self.draw_ids:
                    label = f"ID:{det.track_id} {det.class_name} {det.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)
                    cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                               self.text_scale, (255, 255, 255), self.text_thickness)

        # 2. Рисуем репроекции 3D треков
        if self.draw_reprojection and tracks_3d:
            cam = self.camera_manager.get_camera(cam_id)
            for track_id, track in tracks_3d.items():
                if not track.is_active():
                    continue
                pos_3d = track.get_position()
                px = cam.project_world_to_image(pos_3d, detections.frame_idx)
                if px:
                    u, v = int(px[0]), int(px[1])
                    color = self._get_color(track_id)

                    # Круг репроекции
                    cv2.circle(vis, (u, v), 8, color, 2)
                    cv2.circle(vis, (u, v), 3, color, -1)

                    # ID трека
                    if self.draw_ids:
                        label = f"3D:{track_id}"
                        cv2.putText(vis, label, (u + 10, v - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, color, self.text_thickness)

                    # Траил репроекций
                    if self.draw_trails:
                        if track_id not in self._reprojection_history:
                            self._reprojection_history[track_id] = []
                        self._reprojection_history[track_id].append((u, v))
                        if len(self._reprojection_history[track_id]) > self.max_trail_length:
                            self._reprojection_history[track_id].pop(0)

                        trail = self._reprojection_history[track_id]
                        for k in range(1, len(trail)):
                            alpha = k / len(trail)
                            c = tuple(int(c * alpha) for c in color)
                            cv2.line(vis, trail[k-1], trail[k], c, 2)

        return vis

    def clear_history(self):
        self._reprojection_history.clear()


# =============================================================================
# 3D Trajectory Visualizer
# =============================================================================

class Visualizer3D:
    """Создание 3D анимации траекторий и позиций камер."""

    def __init__(
        self,
        camera_manager: CameraManager,
        output_size: Tuple[int, int] = (1200, 900),
        colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        self.camera_manager = camera_manager
        self.output_size = output_size
        self.colors = colors or [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

    def create_animation(
        self,
        tracks_history: Dict[int, List[np.ndarray]],  # track_id -> list of positions per frame
        output_path: str,
        fps: int = 30,
        max_frames: int = None,
        show_cameras: bool = True,
        show_ground: bool = True
    ):
        """Создаёт 3D GIF/видео анимацию траекторий."""
        fig = plt.figure(figsize=(self.output_size[0]/100, self.output_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Настройка осей
        all_positions = np.vstack([np.array(pos) for pos in tracks_history.values() if len(pos) > 0])
        if len(all_positions) > 0:
            margin = 1000  # 10 метров
            x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
            y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
            z_min, z_max = all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title('3D Trajectories')

        # Рисуем камеры
        if show_cameras:
            self._draw_cameras(ax)

        # Линии траекторий
        track_lines = {}
        track_points = {}
        for track_id, positions in tracks_history.items():
            if len(positions) == 0:
                continue
            color = np.array(self.colors[track_id % len(self.colors)]) / 255.0
            line, = ax.plot([], [], [], color=color, linewidth=2, label=f'Track {track_id}')
            point, = ax.plot([], [], [], 'o', color=color, markersize=8)
            track_lines[track_id] = line
            track_points[track_id] = point

        ax.legend()

        # Фреймы для анимации
        max_len = max(len(p) for p in tracks_history.values()) if tracks_history else 0
        if max_frames:
            max_len = min(max_len, max_frames)

        def animate(frame_idx):
            for track_id, positions in tracks_history.items():
                if len(positions) == 0:
                    continue
                pos = np.array(positions[:frame_idx+1])
                if len(pos) > 0:
                    track_lines[track_id].set_data(pos[:, 0], pos[:, 1])
                    track_lines[track_id].set_3d_properties(pos[:, 2])
                    track_points[track_id].set_data([pos[-1, 0]], [pos[-1, 1]])
                    track_points[track_id].set_3d_properties([pos[-1, 2]])
            return list(track_lines.values()) + list(track_points.values())

        anim = animation.FuncAnimation(fig, animate, frames=max_len, interval=1000/fps, blit=True)
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
        print(f"[Visualizer3D] Сохранена анимация: {output_path}")

    def _draw_cameras(self, ax):
        """Рисует позиции и направления камер."""
        for cam in self.camera_manager.cameras.values():
            pos = cam.extrinsics.location
            # Направление взгляда (forward = column 0 of R)
            R = cam.extrinsics.R_cam_to_world
            forward = R[:, 0] * 2000  # 20 метров вперёд

            # Позиция камеры
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c='black', s=100, marker='^', depthshade=False)
            # Направление
            ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2],
                     color='black', arrow_length_ratio=0.1, linewidth=2)
            # Лейбл
            ax.text(pos[0], pos[1], pos[2] + 500, f'Cam{cam.id}', fontsize=8)

    def create_static_plot(
        self,
        tracks_history: Dict[int, List[np.ndarray]],
        output_path: str,
        show_cameras: bool = True
    ):
        """Создаёт статичный 3D график всех траекторий."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        for track_id, positions in tracks_history.items():
            if len(positions) == 0:
                continue
            pos = np.array(positions)
            color = np.array(self.colors[track_id % len(self.colors)]) / 255.0
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=2, label=f'Track {track_id}')
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, s=100, marker='o')
            ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, s=100, marker='s')

        if show_cameras:
            self._draw_cameras(ax)

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title('3D Trajectories (Static)')
        ax.legend()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer3D] Сохранён статичный график: {output_path}")


# =============================================================================
# Plots Visualizer (X/Y/Z vs time, errors, etc.)
# =============================================================================

class VisualizerPlots:
    """Создание 2D графиков: координаты от времени, ошибки, метрики."""

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_trajectories_vs_time(
        self,
        tracks: Dict[int, Track3D],
        fps: float = 30.0,
        output_path: str = None
    ):
        """Графики X(t), Y(t), Z(t) для каждого трека."""
        if output_path is None:
            output_path = self.output_dir / "trajectories_vs_time.png"

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].set_ylabel('X (cm)')
        axes[1].set_ylabel('Y (cm)')
        axes[2].set_ylabel('Z (cm)')
        axes[2].set_xlabel('Time (s)')
        fig.suptitle('3D Trajectories vs Time')

        for track_id, track in tracks.items():
            if not track.history_3d:
                continue
            pos = np.array(track.history_3d)
            times = np.arange(len(pos)) / fps
            color = np.random.rand(3,)

            axes[0].plot(times, pos[:, 0], label=f'Track {track_id}', color=color, alpha=0.7)
            axes[1].plot(times, pos[:, 1], color=color, alpha=0.7)
            axes[2].plot(times, pos[:, 2], color=color, alpha=0.7)

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[VisualizerPlots] Сохранён график траекторий: {output_path}")

    def plot_velocity_vs_time(
        self,
        tracks: Dict[int, Track3D],
        fps: float = 30.0,
        output_path: str = None
    ):
        """Графики скоростей от времени."""
        if output_path is None:
            output_path = self.output_dir / "velocity_vs_time.png"

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].set_ylabel('Vx (cm/s)')
        axes[1].set_ylabel('Vy (cm/s)')
        axes[2].set_ylabel('Vz (cm/s)')
        axes[2].set_xlabel('Time (s)')
        fig.suptitle('Velocities vs Time')

        for track_id, track in tracks.items():
            if not track.history_3d:
                continue
            pos = np.array(track.history_3d)
            if len(pos) < 2:
                continue
            vel = np.diff(pos, axis=0) * fps
            times = np.arange(len(vel)) / fps
            color = np.random.rand(3,)

            axes[0].plot(times, vel[:, 0], label=f'Track {track_id}', color=color, alpha=0.7)
            axes[1].plot(times, vel[:, 1], color=color, alpha=0.7)
            axes[2].plot(times, vel[:, 2], color=color, alpha=0.7)

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[VisualizerPlots] Сохранён график скоростей: {output_path}")

    def plot_errors(
        self,
        reproj_errors_history: Dict[int, List[float]],
        output_path: str = None
    ):
        """График репроекционных ошибок по трекам."""
        if output_path is None:
            output_path = self.output_dir / "reprojection_errors.png"

        fig, ax = plt.subplots(figsize=(12, 6))
        for track_id, errors in reproj_errors_history.items():
            if errors:
                ax.plot(errors, label=f'Track {track_id}', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Reprojection Error (px)')
        ax.set_title('Reprojection Errors per Track')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[VisualizerPlots] Сохранён график ошибок: {output_path}")

    def plot_metrics_summary(
        self,
        metrics: Dict[str, Any],
        output_path: str = None
    ):
        """Барплот сводных метрик."""
        if output_path is None:
            output_path = self.output_dir / "metrics_summary.png"

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 3D errors
        if 'error_3d_cm' in metrics:
            e3d = metrics['error_3d_cm']
            axes[0, 0].bar(['MAE', 'RMSE', 'Max'], [e3d.get('mean', 0), e3d.get('rmse', 0), e3d.get('max', 0)])
            axes[0, 0].set_ylabel('Error (cm)')
            axes[0, 0].set_title('3D Position Errors')

        # Reprojection errors
        if 'reprojection_error_px' in metrics:
            rep = metrics['reprojection_error_px']
            axes[0, 1].bar(['MAE', 'RMSE', 'Max'], [rep.get('mean', 0), rep.get('rmse', 0), rep.get('max', 0)])
            axes[0, 1].set_ylabel('Error (px)')
            axes[0, 1].set_title('Reprojection Errors')

        # Per-axis errors
        if 'error_3d_cm' in metrics:
            e3d = metrics['error_3d_cm']
            axes[1, 0].bar(['|dX|', '|dY|', '|dZ|'],
                          [e3d.get('mean_abs_dx', 0), e3d.get('mean_abs_dy', 0), e3d.get('mean_abs_dz', 0)])
            axes[1, 0].set_ylabel('Error (cm)')
            axes[1, 0].set_title('Per-Axis 3D Errors')

        # Track stats
        if 'track_stats' in metrics:
            ts = metrics['track_stats']
            axes[1, 1].bar(['Active', 'Confirmed', 'Total Created', 'Deleted'],
                          [ts.get('active_tracks', 0), ts.get('confirmed_tracks', 0),
                           ts.get('total_tracks_created', 0), ts.get('total_tracks_deleted', 0)])
            axes[1, 1].set_title('Track Statistics')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[VisualizerPlots] Сохранены метрики: {output_path}")


# =============================================================================
# Video Writer
# =============================================================================

class VideoWriter:
    """Запись видео с наложениями."""

    def __init__(self, output_path: str, fps: float = 30.0, size: Tuple[int, int] = (1920, 1080)):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, size)
        self.size = size

    def write(self, frame: np.ndarray):
        if frame.shape[:2][::-1] != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        print(f"[VideoWriter] Видео сохранено: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# =============================================================================
# Multi-Camera Visualizer (Combined)
# =============================================================================

class MultiCameraVisualizer:
    """Комбинированный визуализатор для всего пайплайна."""

    def __init__(
        self,
        camera_manager: CameraManager,
        config: Dict
    ):
        self.camera_manager = camera_manager
        self.config = config
        vis_config = config.get('visualization', {})

        self.vis2d = Visualizer2D(
            camera_manager,
            colors=vis_config.get('colors'),
            draw_boxes=vis_config.get('draw_2d_boxes', True),
            draw_reprojection=vis_config.get('draw_3d_reprojection', True),
            max_trail_length=vis_config.get('max_trail_length', 100)
        )
        self.vis3d = Visualizer3D(camera_manager, colors=vis_config.get('colors'))
        self.vis_plots = VisualizerPlots()
        self.video_writers: Dict[int, VideoWriter] = {}

    def init_video_writers(self, output_dir: str, fps: float = 30.0):
        """Инициализирует VideoWriter для каждой камеры."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cam_id in self.camera_manager.get_camera_ids():
            cam = self.camera_manager.get_camera(cam_id)
            out_path = out_dir / f"cam{cam_id}_overlay.mp4"
            self.video_writers[cam_id] = VideoWriter(
                str(out_path),
                fps=fps,
                size=(cam.intrinsics.width, cam.intrinsics.height)
            )

    def process_frame(
        self,
        frame_idx: int,
        frames: Dict[int, np.ndarray],
        detections_per_cam: Dict[int, CameraDetections],
        tracks_3d: Dict[int, Track3D]
    ) -> Dict[int, np.ndarray]:
        """Обрабатывает кадр: рисует overlay на всех камерах."""
        results = {}
        for cam_id, frame in frames.items():
            if frame is None:
                continue
            detections = detections_per_cam.get(cam_id)
            if detections:
                vis_frame = self.vis2d.draw_detections(frame, cam_id, detections, tracks_3d)
                results[cam_id] = vis_frame

                # Запись в видео
                if cam_id in self.video_writers:
                    self.video_writers[cam_id].write(vis_frame)
            else:
                results[cam_id] = frame
        return results

    def finalize(
        self,
        tracks: Dict[int, Track3D],
        output_dir: str,
        reproj_errors_history: Optional[Dict[int, List[float]]] = None,
        metrics: Optional[Dict] = None
    ):
        """Финализация: сохраняет 3D анимацию, графики, метрики."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Закрываем видеоряйтеры
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()

        # 3D анимация
        if self.config.get('visualization', {}).get('draw_trajectory_3d', True):
            tracks_history = {tid: t.history_3d for tid, t in tracks.items() if t.history_3d}
            if tracks_history:
                # GIF
                self.vis3d.create_animation(
                    tracks_history,
                    str(out_dir / "trajectory_3d.gif"),
                    fps=self.config.get('visualization', {}).get('output_fps', 30)
                )
                # Статичный PNG
                self.vis3d.create_static_plot(
                    tracks_history,
                    str(out_dir / "trajectory_3d_static.png")
                )

        # Графики
        if self.config.get('visualization', {}).get('draw_plots', True):
            self.vis_plots.plot_trajectories_vs_time(tracks, output_path=str(out_dir / "trajectories_vs_time.png"))
            self.vis_plots.plot_velocity_vs_time(tracks, output_path=str(out_dir / "velocity_vs_time.png"))
            if reproj_errors_history:
                self.vis_plots.plot_errors(reproj_errors_history, output_path=str(out_dir / "reprojection_errors.png"))
            if metrics:
                self.vis_plots.plot_metrics_summary(metrics, output_path=str(out_dir / "metrics_summary.png"))

        print(f"[MultiCameraVisualizer] Все результаты сохранены в: {out_dir}")


# =============================================================================
# Factory
# =============================================================================

def create_visualizer(camera_manager: CameraManager, config_path: str = None) -> MultiCameraVisualizer:
    """Фабрика для создания визуализатора."""
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return MultiCameraVisualizer(camera_manager, config)


if __name__ == "__main__":
    # Тест
    from camera_manager import create_camera_manager

    cm = create_camera_manager()
    cm.open_all()

    vis = create_visualizer(cm)

    # Тест 2D рисования
    ret, frame = cm.cameras[1].read()
    if ret:
        # Фейковые детекции
        from detector import Detection2D, CameraDetections
        fake_det = Detection2D(
            cam_id=1, frame_idx=0, track_id=0, class_id=0, class_name="truck",
            bbox=np.array([100, 100, 300, 200]), center=np.array([200, 150]), confidence=0.9
        )
        fake_cam_dets = CameraDetections(1, 0, [fake_det])
        vis_frame = vis.vis2d.draw_detections(frame, 1, fake_cam_dets)
        cv2.imwrite("test_vis.jpg", vis_frame)
        print("Тест визуализации OK")

    cm.close_all()