#!/usr/bin/env python
"""
Standalone 3D Track Visualization Script
Создаёт красивый 3D PNG с траекториями движения объектов.

Usage:
    python scripts/plot_3d_tracks.py                    # из metrics.json
    python scripts/plot_3d_tracks.py --tracks output/tracks_history.pkl  # из pickle
    python scripts/plot_3d_tracks.py --cameras config/cameras.json --output my_plot.png
"""

from __future__ import annotations

import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


# =============================================================================
# Data Loading
# =============================================================================

def load_tracks_from_metrics(metrics_path: Path) -> Dict[int, List[np.ndarray]]:
    """
    Загружает 3D траектории из metrics.json (только если есть history).
    В текущем формате metrics.json хранит только сводку, не историю.
    """
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # Если истории нет в JSON, возвращаем пусто
    return {}


def load_tracks_from_pickle(pickle_path: Path) -> Dict[int, List[np.ndarray]]:
    """Загружает tracks_history из pickle файла."""
    with open(pickle_path, 'rb') as f:
        tracks_history = pickle.load(f)

    result = {}
    for track_id, track in tracks_history.items():
        if hasattr(track, 'history_3d') and track.history_3d:
            result[track_id] = [np.array(p) for p in track.history_3d]
    return result


def load_cameras(cameras_json: Path) -> Dict[int, Dict]:
    """Загружает позиции и ориентации камер из cameras.json."""
    with open(cameras_json, 'r', encoding='utf-8') as f:
        config = json.load(f)

    cameras = {}
    for cam_config in config['cameras']:
        cam_id = cam_config['id']
        loc = np.array(cam_config['extrinsics']['location_cm'], dtype=float)
        rot = np.array(cam_config['extrinsics']['rotation_deg'], dtype=float)
        cameras[cam_id] = {
            'location': loc,
            'rotation_deg': rot,
            'name': cam_config.get('name', f'Cam{cam_id}')
        }
    return cameras


def generate_demo_tracks() -> Dict[int, List[np.ndarray]]:
    """Генерирует демо-траектории для теста без данных."""
    np.random.seed(42)
    tracks = {}

    # Грузовик едет по траектории
    n_frames = 200
    t = np.linspace(0, 20, n_frames)

    # Track 1: грузовик едет прямо
    tracks[1] = [
        np.array([500 + 50 * ti, 1000 + 20 * ti, 350 + 10 * np.sin(ti * 0.5)])
        for ti in t
    ]

    # Track 2: поворот
    tracks[2] = [
        np.array([2000 + 30 * ti, 500 + 40 * np.sin(ti * 0.3), 300 + 5 * np.cos(ti * 0.2)])
        for ti in t
    ]

    # Track 3: другой путь
    tracks[3] = [
        np.array([1000 + 25 * ti, 2000 + 30 * ti, 400 + 8 * np.sin(ti * 0.4)])
        for ti in t
    ]

    return tracks


# =============================================================================
# 3D Plotting
# =============================================================================

def create_3d_plot(
    tracks: Dict[int, List[np.ndarray]],
    cameras: Optional[Dict[int, Dict]] = None,
    output_path: Path = Path("trajectory_3d.png"),
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    show_cameras: bool = True,
    show_start_end: bool = True,
    show_ground: bool = True,
    elev: float = 20,
    azim: float = -60,
    trail_alpha: float = 0.7,
    point_size: int = 60
):
    """
    Создаёт красивый 3D график траекторий.

    Args:
        tracks: {track_id: [position_array, ...]} — позиции в см
        cameras: {cam_id: {location, rotation_deg, name}}
        output_path: куда сохранить PNG
        figsize: размер фигуры в дюймах
        dpi: разрешение
        show_cameras: рисовать камеры
        show_start_end: отмечать старт/финиш
        show_ground: рисовать плоскость земли
        elev, azim: угол обзора
        trail_alpha: прозрачность траектории
        point_size: размер маркеров старт/финиш
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Цветовая карта для треков
    n_tracks = len(tracks)
    colors = cm.tab10(np.linspace(0, 1, max(n_tracks, 10)))

    all_positions = []

    # Рисуем траектории
    for i, (track_id, positions) in enumerate(tracks.items()):
        if len(positions) == 0:
            continue

        pos = np.array(positions)  # (N, 3)
        all_positions.append(pos)

        color = colors[i % len(colors)]

        # Линия траектории
        ax.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            color=color, linewidth=2.5, alpha=trail_alpha,
            label=f'Track {track_id} ({len(pos)} pts)'
        )

        # Старт (круг)
        if show_start_end:
            ax.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2],
                color=color, s=point_size, marker='o',
                edgecolors='white', linewidths=1.5, depthshade=False,
                zorder=10
            )
            # Финиш (квадрат)
            ax.scatter(
                pos[-1, 0], pos[-1, 1], pos[-1, 2],
                color=color, s=point_size, marker='s',
                edgecolors='white', linewidths=1.5, depthshade=False,
                zorder=10
            )

            # Направление движения (стрелка в середине)
            mid_idx = len(pos) // 2
            if mid_idx + 5 < len(pos):
                p1 = pos[mid_idx]
                p2 = pos[mid_idx + 5]
                dx, dy, dz = p2 - p1
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 0:
                    ax.quiver(
                        p1[0], p1[1], p1[2],
                        dx/norm*200, dy/norm*200, dz/norm*200,
                        color=color, arrow_length_ratio=0.2, linewidth=2
                    )

    # Камеры
    if show_cameras and cameras:
        for cam_id, cam in cameras.items():
            loc = cam['location']
            name = cam.get('name', f'Cam{cam_id}')

            # Позиция камеры
            ax.scatter(
                loc[0], loc[1], loc[2],
                c='black', s=150, marker='^',
                edgecolors='white', linewidths=1.5,
                depthshade=False, zorder=20,
                label=f'{name} (Cam {cam_id})' if cam_id == 1 else ""
            )

            # Направление взгляда (forward = column 0 of R)
            rot = cam['rotation_deg']
            # UE convention: roll инвертирован
            import math
            r, p, y = map(math.radians, (-rot[0], rot[1], rot[2]))
            cr, sr = math.cos(r), math.sin(r)
            cp, sp = math.cos(p), math.sin(p)
            cy, sy = math.cos(y), math.sin(y)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            forward = R[:, 0] * 3000  # 30 метров вперёд

            ax.quiver(
                loc[0], loc[1], loc[2],
                forward[0], forward[1], forward[2],
                color='black', arrow_length_ratio=0.1,
                linewidth=2, alpha=0.8
            )

            # Лейбл
            ax.text(
                loc[0], loc[1], loc[2] + 500,
                f'Cam{cam_id}',
                fontsize=10, fontweight='bold',
                ha='center', va='bottom'
            )

    # Плоскость земли
    if show_ground and all_positions:
        all_pos = np.vstack(all_positions)
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        z_min = all_pos[:, 2].min()

        margin = 500
        xx, yy = np.meshgrid(
            np.linspace(x_min - margin, x_max + margin, 20),
            np.linspace(y_min - margin, y_max + margin, 20)
        )
        zz = np.full_like(xx, z_min - 100)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray', rstride=2, cstride=2)

    # Настройка осей
    if all_positions:
        all_pos = np.vstack(all_positions)
        margin = 1000
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    ax.set_xlabel('X (cm)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (cm)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (cm)', fontsize=12, labelpad=10)
    ax.set_title('3D Object Trajectories', fontsize=16, fontweight='bold', pad=20)

    # Угол обзора
    ax.view_init(elev=elev, azim=azim)

    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Убираем дубликаты
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(), unique.keys(),
            loc='upper left', fontsize=10,
            framealpha=0.9, fancybox=True
        )

    # Сетка
    ax.grid(True, alpha=0.3)

    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Также сохранить в текущую папку если output не существует

    print(f"[OK] 3D plot saved: {output_path}")
    print(f"       Tracks: {len(tracks)}, Cameras: {len(cameras) if cameras else 0}")


# =============================================================================
# Additional: Time-series plots (X/Y/Z vs time)
# =============================================================================

def create_timeseries_plots(
    tracks: Dict[int, List[np.ndarray]],
    output_path: Path,
    fps: float = 30.0,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150
):
    """Создаёт графики X(t), Y(t), Z(t) и скоростей V(t)."""
    fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=dpi)
    fig.suptitle('Track Motion Analysis', fontsize=16, fontweight='bold')

    colors = cm.tab10(np.linspace(0, 1, 10))

    for i, (track_id, positions) in enumerate(tracks.items()):
        if len(positions) < 2:
            continue

        pos = np.array(positions)
        times = np.arange(len(pos)) / fps
        color = colors[i % len(colors)]

        # Позиции
        axes[0, 0].plot(times, pos[:, 0], color=color, alpha=0.8, label=f'T{track_id}', linewidth=1.5)
        axes[0, 1].plot(times, pos[:, 1], color=color, alpha=0.8, linewidth=1.5)
        axes[1, 0].plot(times, pos[:, 2], color=color, alpha=0.8, linewidth=1.5)

        # Скорости
        vel = np.diff(pos, axis=0) * fps
        vel_times = times[:-1]
        axes[1, 1].plot(vel_times, vel[:, 0], color=color, alpha=0.7, linewidth=1)
        axes[2, 0].plot(vel_times, vel[:, 1], color=color, alpha=0.7, linewidth=1)
        axes[2, 1].plot(vel_times, vel[:, 2], color=color, alpha=0.7, linewidth=1)

    # Подписи
    axes[0, 0].set_ylabel('X (cm)')
    axes[0, 1].set_ylabel('Y (cm)')
    axes[1, 0].set_ylabel('Z (cm)')
    axes[1, 1].set_ylabel('Vx (cm/s)')
    axes[2, 0].set_ylabel('Vy (cm/s)')
    axes[2, 1].set_ylabel('Vz (cm/s)')

    for ax in axes.flat:
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Timeseries plot saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create 3D PNG visualization of track trajectories"
    )
    parser.add_argument(
        '--metrics', '-m', type=Path,
        default=Path('output/metrics.json'),
        help='Path to metrics.json (default: output/metrics.json)'
    )
    parser.add_argument(
        '--tracks', '-t', type=Path,
        help='Path to tracks_history.pkl (if available)'
    )
    parser.add_argument(
        '--cameras', '-c', type=Path,
        default=Path('config/cameras.json'),
        help='Path to cameras.json (default: config/cameras.json)'
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        default=Path('output/trajectory_3d_custom.png'),
        help='Output PNG path (default: output/trajectory_3d_custom.png)'
    )
    parser.add_argument(
        '--timeseries', action='store_true',
        help='Also create X/Y/Z vs time plots'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Use demo data instead of real tracks'
    )
    parser.add_argument(
        '--elev', type=float, default=20,
        help='Elevation angle (default: 20)'
    )
    parser.add_argument(
        '--azim', type=float, default=-60,
        help='Azimuth angle (default: -60)'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='FPS for timeseries (default: 30)'
    )

    args = parser.parse_args()

    # Загрузка треков
    if args.demo:
        print("[INFO] Using demo data...")
        tracks = generate_demo_tracks()
    elif args.tracks and args.tracks.exists():
        print(f"[INFO] Loading tracks from {args.tracks}")
        tracks = load_tracks_from_pickle(args.tracks)
    else:
        print(f"[WARN] No tracks source found. Using demo data.")
        tracks = generate_demo_tracks()

    if not tracks:
        print("[ERROR] No track data available!")
        return 1

    print(f"[INFO] Loaded {len(tracks)} tracks")
    for tid, pts in tracks.items():
        print(f"  Track {tid}: {len(pts)} points")

    # Загрузка камер
    cameras = None
    if args.cameras and args.cameras.exists():
        cameras = load_cameras(args.cameras)
        print(f"[INFO] Loaded {len(cameras)} cameras")

    # 3D траектории
    create_3d_plot(
        tracks=tracks,
        cameras=cameras,
        output_path=args.output,
        elev=args.elev,
        azim=args.azim
    )

    # Time-series plots
    if args.timeseries:
        ts_output = args.output.with_stem(args.output.stem + '_timeseries')
        create_timeseries_plots(
            tracks=tracks,
            output_path=ts_output,
            fps=args.fps
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())