"""
Pipeline — главное объединение всех компонентов в единый пайплайн.
Асинхронная обработка, сбор статистики, управление ресурсами.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import numpy as np
import yaml

from .camera_manager import CameraManager, create_camera_manager
from .detector import MultiCameraDetector, create_multi_camera_detector, CameraDetections, Detection2D
from .triangulator import Triangulator, create_triangulator, TriangulationResult
from .associator import HybridAssociator, create_associator
from .tracker3d import MultiObjectTracker3D, create_tracker3d, Track3D
from .visualizer import MultiCameraVisualizer, create_visualizer
from .monocular_depth import create_monocular_depth_estimator, MultiCameraMonocularDepth, DepthEstimate

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PipelineStats:
    """Статистика выполнения пайплайна."""
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    total_detections_2d: int = 0
    total_triangulations: int = 0
    successful_triangulations: int = 0
    total_tracks_3d: int = 0
    active_tracks_3d: int = 0

    # Timing (ms)
    detection_time_ms: float = 0.0
    triangulation_time_ms: float = 0.0
    association_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    visualization_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Per-frame timing
    frame_times: List[float] = field(default_factory=list)

    @property
    def avg_fps(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return (self.processed_frames * 1000.0) / self.total_time_ms

    @property
    def avg_frame_time_ms(self) -> float:
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times)

    def to_dict(self) -> Dict:
        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "total_detections_2d": self.total_detections_2d,
            "total_triangulations": self.total_triangulations,
            "successful_triangulations": self.successful_triangulations,
            "total_tracks_3d": self.total_tracks_3d,
            "active_tracks_3d": self.active_tracks_3d,
            "avg_fps": round(self.avg_fps, 2),
            "avg_frame_time_ms": round(self.avg_frame_time_ms, 2),
            "detection_time_ms": round(self.detection_time_ms, 2),
            "triangulation_time_ms": round(self.triangulation_time_ms, 2),
            "association_time_ms": round(self.association_time_ms, 2),
            "tracking_time_ms": round(self.tracking_time_ms, 2),
            "visualization_time_ms": round(self.visualization_time_ms, 2),
        }


@dataclass
class FrameData:
    """Данные одного кадра для передачи между этапами."""
    frame_idx: int
    timestamp: float
    frames: Dict[int, np.ndarray]           # cam_id -> frame
    detections: Dict[int, CameraDetections]  # cam_id -> detections
    triangulation_results: Optional[List[TriangulationResult]] = None
    associations: Optional[Dict] = None      # associations info
    tracks_3d: Optional[Dict[int, Track3D]] = None


# =============================================================================
# Pipeline Stages
# =============================================================================

class PipelineStage:
    """Базовый класс этапа пайплайна."""

    def __init__(self, name: str):
        self.name = name
        self.stats = {"calls": 0, "total_time_ms": 0.0, "errors": 0}

    def process(self, frame_data: FrameData) -> FrameData:
        raise NotImplementedError

    def _time_it(self, func: Callable, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        self.stats["calls"] += 1
        self.stats["total_time_ms"] += elapsed
        self.stats["last_elapsed_ms"] = elapsed
        return result, elapsed


class DetectionStage(PipelineStage):
    """Этап детекции на всех камерах."""

    def __init__(self, detector: MultiCameraDetector):
        super().__init__("Detection")
        self.detector = detector

    def _filter_detections(self, detections_per_cam: Dict[int, CameraDetections]) -> Dict[int, CameraDetections]:
        det_config = self.detector.config.get('detection', {})
        classes = det_config.get('classes')
        min_area = det_config.get('min_detection_area', 0)
        min_conf = det_config.get('conf_threshold', 0.25)
        result = {}
        for cam_id, cam_dets in detections_per_cam.items():
            filtered = []
            for det in cam_dets.detections:
                if det.confidence < min_conf:
                    continue
                if min_area > 0 and det.area < min_area:
                    continue
                if classes is not None:
                    if det.class_id not in classes and det.class_name not in classes:
                        continue
                filtered.append(det)
            result[cam_id] = CameraDetections(
                cam_id, cam_dets.frame_idx, filtered, cam_dets.timestamp
            )
        return result

    def process(self, frame_data: FrameData) -> FrameData:
        detections, elapsed = self._time_it(
            self.detector.process_all,
            frame_data.frames,
            frame_data.frame_idx,
            frame_data.timestamp
        )
        detections = self._filter_detections(detections)
        frame_data.detections = detections
        return frame_data


class TriangulationStage(PipelineStage):
    """Этап триангуляции 3D точек.

    Использует кросс-камерные группы из AssociationStage (каждая группа уже
    содержит TriangulationResult). Если ассоциаций нет — откатывается на
    прямую триангуляцию детекций.
    """

    def __init__(self, triangulator: Triangulator):
        super().__init__("Triangulation")
        self.triangulator = triangulator

    def process(self, frame_data: FrameData) -> FrameData:
        associations = getattr(frame_data, "associations", None) or []
        if associations:
            results = [
                g.triangulation
                for g in associations
                if g.triangulation is not None and g.triangulation.success
            ]
            frame_data.triangulation_results = results
            self.stats["calls"] += 1
            self.stats["total_time_ms"] += 0.0
            self.stats["last_elapsed_ms"] = 0.0
            return frame_data

        detections_dict = {
            cam_id: cam_dets.detections
            for cam_id, cam_dets in frame_data.detections.items()
        }
        results, elapsed = self._time_it(
            self.triangulator.triangulate,
            detections_dict,
            frame_data.frame_idx
        )
        frame_data.triangulation_results = results
        return frame_data


class MonocularDepthStage(PipelineStage):
    """Этап моноскулрной оценки глубины для одиночных камер.

    Когда триангуляция невозможна (менее 2 камер видят объект),
    использует размер бокса + известные размеры грузовика для оценки Z.
    Добавляет моноскулрные 3D точки к результатам триангуляции.
    """

    def __init__(self, mono_depth: MultiCameraMonocularDepth, enabled: bool = True):
        super().__init__("MonocularDepth")
        self.mono_depth = mono_depth
        self.enabled = enabled

    def process(self, frame_data: FrameData) -> FrameData:
        if not self.enabled:
            return frame_data

        existing_triang = frame_data.triangulation_results or []

        detections_dict = {
            cam_id: cam_dets.detections
            for cam_id, cam_dets in frame_data.detections.items()
        }

        mono_results = self.mono_depth.process_single_camera_detections(
            detections_dict, frame_data.frame_idx
        )

        from types import SimpleNamespace
        for cam_id, cam_results in mono_results.items():
            for track_id, point_3d, depth_est in cam_results:
                mono_tri = SimpleNamespace(
                    point_3d=point_3d,
                    success=True,
                    method="monocular",
                    cam_id=cam_id,
                    track_id=track_id,
                    covariance=np.eye(3) * (1000.0 / max(depth_est.confidence, 0.1))**2,
                    depth_estimate=depth_est
                )
                existing_triang.append(mono_tri)

        frame_data.triangulation_results = existing_triang
        return frame_data


class AssociationStage(PipelineStage):
    """Этап ассоциации между камерами."""

    def __init__(self, associator: HybridAssociator):
        super().__init__("Association")
        self.associator = associator

    def process(self, frame_data: FrameData) -> FrameData:
        # Extract detection lists for associator
        detections_dict = {
            cam_id: cam_dets.detections
            for cam_id, cam_dets in frame_data.detections.items()
        }
        associations, elapsed = self._time_it(
            self.associator.associate,
            frame_data.frames,
            detections_dict,
            frame_data.frame_idx
        )
        frame_data.associations = associations
        return frame_data


class Tracking3DStage(PipelineStage):
    """Этап 3D трекинга."""

    def __init__(self, tracker: MultiObjectTracker3D):
        super().__init__("Tracking3D")
        self.tracker = tracker

    def process(self, frame_data: FrameData) -> FrameData:
        tracks, elapsed = self._time_it(
            self.tracker.step,
            frame_data.triangulation_results,
            frame_data.frame_idx
        )
        frame_data.tracks_3d = tracks
        return frame_data


class VisualizationStage(PipelineStage):
    """Этап визуализации."""

    def __init__(self, visualizer: MultiCameraVisualizer, enabled: bool = True):
        super().__init__("Visualization")
        self.visualizer = visualizer
        self.enabled = enabled

    def process(self, frame_data: FrameData) -> FrameData:
        if not self.enabled:
            return frame_data
        _, elapsed = self._time_it(
            self.visualizer.process_frame,
            frame_data.frame_idx,
            frame_data.frames,
            frame_data.detections,
            frame_data.tracks_3d or {}
        )
        return frame_data


# =============================================================================
# Main Pipeline
# =============================================================================

class MultiCameraTrackingPipeline:
    """
    Главный класс пайплайна multi-camera 3D tracking.
    Поддерживает синхронную и асинхронную обработку.
    """

    def __init__(self, config: Dict, config_path: str = None):
        self.config = config
        self.config_path = config_path
        self.pipeline_config = config.get('pipeline', {})
        self.vis_config = config.get('visualization', {})

        # Компоненты
        self.camera_manager: Optional[CameraManager] = None
        self.detector: Optional[MultiCameraDetector] = None
        self.triangulator: Optional[Triangulator] = None
        self.associator: Optional[HybridAssociator] = None
        self.tracker_3d: Optional[MultiObjectTracker3D] = None
        self.visualizer: Optional[MultiCameraVisualizer] = None

        # Этапы
        self.stages: List[PipelineStage] = []

        # Статистика
        self.stats = PipelineStats()

        # Состояние
        self.running = False
        self.frame_idx = 0
        self.start_time = 0.0

        # История для анализа
        self.reproj_errors_history: Dict[int, List[float]] = {}
        self.tracks_history: Dict[int, Track3D] = {}

        # Thread pool для асинхронной обработки
        self._executor: Optional[ThreadPoolExecutor] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._result_queue: queue.Queue = queue.Queue(maxsize=10)
        self._worker_thread: Optional[threading.Thread] = None

    def initialize(self):
        """Инициализирует все компоненты пайплайна."""
        logger.info("Инициализация пайплайна...")

        # 1. Camera Manager
        self.camera_manager = create_camera_manager()
        self.camera_manager.open_all()
        logger.info(f"Камер открыто: {len(self.camera_manager.get_camera_ids())}")

        # 2. Detector
        self.detector = create_multi_camera_detector(self.camera_manager, self.config_path)

        # 3. Triangulator
        self.triangulator = create_triangulator(self.camera_manager, self.config_path)

        # 3.5. Monocular Depth Estimator
        self.mono_depth = create_monocular_depth_estimator(self.camera_manager, self.config_path)

        # 4. Associator
        self.associator = create_associator(self.camera_manager, self.triangulator, self.config_path)

        # 5. 3D Tracker
        self.tracker_3d = create_tracker3d(self.config_path)

        # 6. Visualizer
        self.visualizer = create_visualizer(self.camera_manager, self.config_path)
        if self.vis_config.get('save_video', False):
            output_dir = self.vis_config.get('output_dir', 'output/video')
            fps = self.vis_config.get('output_fps', 30)
            self.visualizer.init_video_writers(output_dir, fps)

        # 7. Этапы пайплайна
        # Ассоциация должна идти ДО триангуляции: триангулятор опирается на
        # кросс-камерные группы ассоциатора (per-camera track_id не globals).
        # MonocularDepthStage после триангуляции для одиночных камер.
        self.stages = [
            DetectionStage(self.detector),
            AssociationStage(self.associator),
            TriangulationStage(self.triangulator),
            MonocularDepthStage(self.mono_depth, self.config.get('monocular_depth', {}).get('enabled', True)),
            Tracking3DStage(self.tracker_3d),
            VisualizationStage(self.visualizer, self.vis_config.get('enabled', True))
        ]

        logger.info("Пайплайн инициализирован успешно")

    def run_sync(self, max_frames: int = -1, progress_callback: Optional[Callable] = None) -> PipelineStats:
        """
        Синхронный запуск пайплайна.
        Args:
            max_frames: макс. количество кадров (-1 = все)
            progress_callback: callback(frame_idx, total_frames, stats)
        """
        self.running = True
        self.start_time = time.perf_counter()
        self.frame_idx = 0

        total_frames = self._estimate_total_frames()
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Запуск синхронного пайплайна. Всего кадров: {total_frames}")

        try:
            while self.running and (max_frames == -1 or self.frame_idx < max_frames):
                frame_start = time.perf_counter()

                # Читаем кадры
                frames = self.camera_manager.read_all()
                if not any(f is not None for f in frames.values()):
                    logger.info("Конец видео")
                    break

                # Создаём FrameData
                frame_data = FrameData(
                    frame_idx=self.frame_idx,
                    timestamp=self.frame_idx / 30.0,  # предполагаем 30 FPS
                    frames=frames,
                    detections={}
                )

                # Прогоняем через все этапы
                for stage in self.stages:
                    if not self.running:
                        break
                    frame_data = stage.process(frame_data)

                # Собираем статистику
                self._collect_stats(frame_data)

                # Прогресс
                frame_time = (time.perf_counter() - frame_start) * 1000
                self.stats.frame_times.append(frame_time)
                self.stats.total_time_ms += frame_time
                self.stats.total_frames += 1  # Increment total frames

                if progress_callback:
                    progress_callback(self.frame_idx, total_frames, self.stats)

                self.frame_idx += 1
                self.stats.processed_frames += 1

        except KeyboardInterrupt:
            logger.info("Прервано пользователем")
        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}", exc_info=True)
        finally:
            self.running = False
            self._finalize()

        return self.stats

    def run_async(self, max_frames: int = -1, num_workers: int = 2) -> PipelineStats:
        """
        Асинхронный запуск с ThreadPoolExecutor.
        Чтение кадров и детекция в одном потоке, остальные этапы в пуле.
        """
        self.running = True
        self.start_time = time.perf_counter()
        self.frame_idx = 0

        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._worker_thread = threading.Thread(target=self._async_worker)
        self._worker_thread.start()

        total_frames = self._estimate_total_frames()
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Запуск асинхронного пайплайна. Всего кадров: {total_frames}, workers: {num_workers}")

        try:
            # Producer: читает кадры и кладёт в очередь
            while self.running and (max_frames == -1 or self.frame_idx < max_frames):
                frames = self.camera_manager.read_all()
                if not any(f is not None for f in frames.values()):
                    break

                frame_data = FrameData(
                    frame_idx=self.frame_idx,
                    timestamp=self.frame_idx / 30.0,
                    frames=frames,
                    detections={}
                )

                try:
                    self._frame_queue.put(frame_data, timeout=1.0)
                except queue.Full:
                    logger.warning("Очередь полна, пропуск кадра")
                    self.stats.skipped_frames += 1

                self.frame_idx += 1
                self.stats.total_frames += 1

            # Сигнал завершения
            self._frame_queue.put(None)
            self._worker_thread.join(timeout=30.0)

        except KeyboardInterrupt:
            logger.info("Прервано пользователем")
        except Exception as e:
            logger.error(f"Ошибка в асинхронном пайплайне: {e}", exc_info=True)
        finally:
            self.running = False
            self._executor.shutdown(wait=True)
            self._finalize()

        return self.stats

    def _async_worker(self):
        """Worker thread для асинхронной обработки."""
        while self.running:
            try:
                frame_data = self._frame_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                frame_start = time.perf_counter()

                # Прогоняем через этапы (кроме визуализации)
                for stage in self.stages[:-1]:  # без визуализации
                    if not self.running:
                        break
                    frame_data = stage.process(frame_data)

                # Визуализация в главном потоке (OpenCV не thread-safe)
                self._result_queue.put((frame_data, frame_start))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Async worker error: {e}")
                continue

        # Обработка оставшихся результатов визуализации
        while not self._result_queue.empty():
            frame_data, frame_start = self._result_queue.get()
            self.stages[-1].process(frame_data)  # визуализация
            self._collect_stats(frame_data)

    def _collect_stats(self, frame_data: FrameData):
        """Собирает статистику после обработки кадра."""
        self.stats.total_detections_2d += sum(
            len(d.detections) for d in frame_data.detections.values()
        )

        if frame_data.triangulation_results:
            self.stats.total_triangulations += len(frame_data.triangulation_results)
            self.stats.successful_triangulations += sum(
                1 for r in frame_data.triangulation_results if r.success
            )

        if frame_data.tracks_3d:
            self.stats.total_tracks_3d = max(
                self.stats.total_tracks_3d,
                max(frame_data.tracks_3d.keys(), default=0)
            )
            self.stats.active_tracks_3d = sum(
                1 for t in frame_data.tracks_3d.values() if t.is_active()
            )

            # Обновляем историю треков
            for tid, track in frame_data.tracks_3d.items():
                self.tracks_history[tid] = track

            # Репроекционные ошибки
            for tid, track in frame_data.tracks_3d.items():
                if track.reprojection_errors:
                    if tid not in self.reproj_errors_history:
                        self.reproj_errors_history[tid] = []
                    self.reproj_errors_history[tid].extend(track.reprojection_errors)

        # Timing из этапов (берём время последнего кадра, а не накопленное)
        for stage in self.stages:
            if stage.name == "Detection":
                self.stats.detection_time_ms += stage.stats.get("last_elapsed_ms", 0.0)
            elif stage.name == "Triangulation":
                self.stats.triangulation_time_ms += stage.stats.get("last_elapsed_ms", 0.0)
            elif stage.name == "Association":
                self.stats.association_time_ms += stage.stats.get("last_elapsed_ms", 0.0)
            elif stage.name == "MonocularDepth":
                # Можно добавить отдельное поле в stats если нужно
                pass
            elif stage.name == "Tracking3D":
                self.stats.tracking_time_ms += stage.stats.get("last_elapsed_ms", 0.0)
            elif stage.name == "Visualization":
                self.stats.visualization_time_ms += stage.stats.get("last_elapsed_ms", 0.0)

    def _estimate_total_frames(self) -> int:
        """Оценивает общее количество кадров."""
        min_frames = float('inf')
        for cam in self.camera_manager.cameras.values():
            if cam.total_frames > 0:
                min_frames = min(min_frames, cam.total_frames)
        return int(min_frames) if min_frames != float('inf') else 10000

    def _finalize(self):
        """Финализация: сохраняет результаты, закрывает ресурсы."""
        logger.info("Финализация пайплайна...")

        # Метрики трекера
        tracker_metrics = {}
        if self.tracker_3d:
            tracker_metrics = self.tracker_3d.get_stats()

        # Метрики ассоциатора
        associator_metrics = {}
        if self.associator:
            associator_metrics = self.associator.get_stats()

        # Сводные метрики
        metrics = {
            "error_3d_cm": self._compute_3d_errors(),
            "reprojection_error_px": self._compute_reproj_errors(),
            "track_stats": tracker_metrics,
            "association_stats": associator_metrics,
            "pipeline_stats": self.stats.to_dict()
        }

        # Визуализация финальных результатов
        if self.visualizer:
            self.visualizer.finalize(
                self.tracks_history,
                self.vis_config.get('output_dir', 'output'),
                self.reproj_errors_history,
                metrics
            )

        # Закрываем камеры
        if self.camera_manager:
            self.camera_manager.close_all()

        # Логируем итоговую статистику
        logger.info(f"=== ИТОГОВАЯ СТАТИСТИКА ===")
        for k, v in self.stats.to_dict().items():
            logger.info(f"  {k}: {v}")

        # Сохраняем метрики в JSON
        import json
        out_dir = Path(self.vis_config.get('output_dir', 'output'))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Метрики сохранены: {out_dir / 'metrics.json'}")

    def _compute_3d_errors(self) -> Dict[str, float]:
        """Вычисляет 3D ошибки (если есть GT)."""
        # Заглушка - можно расширить при наличии GT
        return {
            "mean": 0.0,
            "rmse": 0.0,
            "max": 0.0,
            "mean_abs_dx": 0.0,
            "mean_abs_dy": 0.0,
            "mean_abs_dz": 0.0
        }

    def _compute_reproj_errors(self) -> Dict[str, float]:
        """Вычисляет репроекционные ошибки."""
        all_errors = []
        for errors in self.reproj_errors_history.values():
            all_errors.extend(errors)
        if not all_errors:
            return {"mean": 0.0, "rmse": 0.0, "max": 0.0}
        all_errors = np.array(all_errors)
        return {
            "mean": float(np.mean(all_errors)),
            "rmse": float(np.sqrt(np.mean(all_errors**2))),
            "max": float(np.max(all_errors))
        }

    def stop(self):
        """Останавливает пайплайн."""
        self.running = False

    def get_current_tracks(self) -> Dict[int, Track3D]:
        """Возвращает текущие активные 3D треки."""
        if self.tracker_3d:
            return {tid: t for tid, t in self.tracker_3d.tracks.items() if t.is_active()}
        return {}


# =============================================================================
# Factory
# =============================================================================

def create_pipeline(config_path: str = None) -> MultiCameraTrackingPipeline:
    """Фабрика для создания пайплайна."""
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return MultiCameraTrackingPipeline(config, config_path)


if __name__ == "__main__":
    # Тест запуска
    pipeline = create_pipeline()
    pipeline.initialize()

    # Запуск на 10 кадрах для теста
    stats = pipeline.run_sync(max_frames=10)

    print(f"\nТест завершён. FPS: {stats.avg_fps:.1f}")
    print(f"Детекций: {stats.total_detections_2d}")
    print(f"Триангуляций: {stats.successful_triangulations}/{stats.total_triangulations}")
    print(f"Активных 3D треков: {stats.active_tracks_3d}")