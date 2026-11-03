#!/usr/bin/env python
"""
Peleng Multi-Camera 3D Tracking Pipeline - CLI Entry Point

Usage:
    python run.py                    # Run with default config
    python run.py --config custom.yaml
    python run.py --max-frames 100 --async
    python run.py --help
"""

import argparse
import sys
import time
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import create_pipeline, MultiCameraTrackingPipeline
from src.camera_manager import create_camera_manager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Peleng Multi-Camera 3D Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Run with default config (tracker.yaml)
  python run.py -c config/my_config.yaml   # Custom config
  python run.py --max-frames 100          # Process only 100 frames
  python run.py --async --workers 4       # Async mode with 4 workers
  python run.py --no-vis                  # Disable visualization
  python run.py --list-cameras            # List available cameras
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: config/tracker.yaml)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='Maximum frames to process (-1 = all)'
    )

    parser.add_argument(
        '--async', '--async-mode',
        action='store_true',
        dest='async_mode',
        help='Run in async mode with thread pool'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of worker threads for async mode (default: 2)'
    )

    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Disable visualization (faster processing)'
    )

    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def list_cameras():
    """Список доступных камер."""
    print("Поиск камер...")
    cm = create_camera_manager()
    cm.open_all()

    print(f"\nНайдено камер: {len(cm.cameras)}")
    for cam_id, cam in cm.cameras.items():
        print(f"\n  Camera {cam_id}:")
        print(f"    Video: {cam.source}")
        print(f"    Resolution: {cam.intrinsics.width}x{cam.intrinsics.height}")
        print(f"    FPS: {cam.fps:.1f}")
        print(f"    Total frames: {cam.total_frames}")
        print(f"    Location (cm): {cam.extrinsics.location}")
        print(f"    Rotation (deg): {cam.extrinsics.rotation_deg}")

    cm.close_all()


def progress_callback(frame_idx: int, total_frames: int, stats):
    """Callback для отображения прогресса."""
    if frame_idx % 30 == 0:  # Каждую секунду при 30 FPS
        pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        print(f"\r  Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
              f"FPS: {stats.avg_fps:.1f} | "
              f"Dets: {stats.total_detections_2d} | "
              f"3D Tracks: {stats.active_tracks_3d}", end='', flush=True)


def main():
    args = parse_args()

    # Настройка логирования
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Список камер и выход
    if args.list_cameras:
        list_cameras()
        return 0

    print("=" * 60)
    print("Peleng Multi-Camera 3D Tracking Pipeline")
    print("=" * 60)

    # Создание пайплайна
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "tracker.yaml"

    print(f"\nЗагрузка конфига: {config_path}")
    pipeline = create_pipeline(str(config_path))

    # Переопределение output_dir если задан
    if args.output_dir:
        pipeline.vis_config['output_dir'] = args.output_dir

    # Отключение визуализации
    if args.no_vis:
        pipeline.vis_config['enabled'] = False
        pipeline.vis_config['save_video'] = False
        pipeline.vis_config['draw_plots'] = False

    # Инициализация
    print("Инициализация компонентов...")
    start_init = time.time()
    pipeline.initialize()
    print(f"Инициализация завершена за {time.time() - start_init:.2f} сек")

    # Запуск
    print(f"\nЗапуск пайплайна ({'async' if args.async_mode else 'sync'} mode)...")
    if args.max_frames > 0:
        print(f"Макс. кадров: {args.max_frames}")

    start_run = time.time()

    try:
        if args.async_mode:
            stats = pipeline.run_async(max_frames=args.max_frames, num_workers=args.workers)
        else:
            stats = pipeline.run_sync(max_frames=args.max_frames, progress_callback=progress_callback)

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C)")
        pipeline.stop()
        return 130
    except Exception as e:
        print(f"\n\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start_run
    print(f"\n\nЗавершено за {elapsed:.2f} сек")

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    for k, v in stats.to_dict().items():
        print(f"  {k}: {v}")

    print(f"\nРезультаты сохранены в: {pipeline.vis_config.get('output_dir', 'output')}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())