"""
monocular_depth.py
============================================================
Моноскулярная оценка глубины для одиночных камер.

Когда только 1 камера видит объект (нет перекрытия камер),
используем размеры бокса + известные размеры грузовика для оценки Z.

Также использует временную согласованность (Kalman Filter 1D по Z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2


@dataclass
class TruckDimensions:
    """Размеры грузовика (примерные для стандартного тягача)."""
    length_cm: float = 650.0    # длина (кабина + прицеп)
    width_cm: float = 250.0     # ширина
    height_cm: float = 350.0    # высота кабины
    # Для бокса детекции (примерный aspect ratio)
    bbox_aspect_ratio: float = 1.5  # width / height


@dataclass
class DepthEstimate:
    """Результат оценки глубины."""
    z_cm: float
    confidence: float
    method: str  # "bbox_size", "temporal", "fusion"
    bbox_width_px: float
    bbox_height_px: float


class MonocularDepthEstimator:
    """
    Оценка глубины (Z координаты) из размера бокса на одной камере.

    Использует:
    1. Размер бокса + известные размеры грузовика + фокусное расстояние
    2. Временную согласованность (1D Kalman Filter по Z)
    3. Фьюжн нескольких оценок (если есть несколько детекций)
    """

    def __init__(
        self,
        camera_manager,
        truck_dims: TruckDimensions = None,
        config: dict = None
    ):
        self.camera_manager = camera_manager
        self.truck_dims = truck_dims or TruckDimensions()
        self.config = config or {}

        # Параметры
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.max_z_cm = self.config.get("max_z_cm", 20000.0)
        self.min_z_cm = self.config.get("min_z_cm", 500.0)

        # 1D Kalman Filter для Z на каждую камеру + трек
        self.kalman_filters: Dict[Tuple[int, int], cv2.KalmanFilter] = {}

        # История для сглаживания
        self.z_history: Dict[Tuple[int, int], List[float]] = {}
        self.max_history = 10

        # Кэш фокусных расстояний в пикселях
        self._focal_cache: Dict[int, Tuple[float, float]] = {}

    def _get_focal_px(self, cam_id: int) -> Tuple[float, float]:
        """Возвращает фокусное расстояние в пикселях (fx, fy)."""
        if cam_id not in self._focal_cache:
            cam = self.camera_manager.get_camera(cam_id)
            fx = cam.intrinsics.K[0, 0]
            fy = cam.intrinsics.K[1, 1]
            self._focal_cache[cam_id] = (fx, fy)
        return self._focal_cache[cam_id]

    def _get_or_create_kf(self, cam_id: int, track_id: int) -> cv2.KalmanFilter:
        """Создаёт или возвращает Kalman Filter для (cam_id, track_id)."""
        key = (cam_id, track_id)
        if key not in self.kalman_filters:
            # State: [z, z_dot]
            # Measurement: z
            kf = cv2.KalmanFilter(2, 1)
            kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)  # dt=1 frame
            kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
            kf.processNoiseCov = np.array([[100, 0], [0, 10]], dtype=np.float32)
            kf.measurementNoiseCov = np.array([[500]], dtype=np.float32)
            kf.errorCovPost = np.array([[1000, 0], [0, 100]], dtype=np.float32)
            self.kalman_filters[key] = kf
            self.z_history[key] = []
        return self.kalman_filters[key]

    def estimate_from_bbox(
        self,
        cam_id: int,
        bbox: np.ndarray,  # [x1, y1, x2, y2]
        track_id: int = -1
    ) -> Optional[DepthEstimate]:
        """
        Оценивает глубину по размеру бокса.

        Формула: Z = (f * H_real) / h_pixels
        где H_real - реальная высота объекта, h_pixels - высота бокса в пикселях.
        """
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox[:4]
        w_px = max(1.0, x2 - x1)
        h_px = max(1.0, y2 - y1)

        if w_px < 10 or h_px < 10:
            return None  # Слишком маленький бокс

        fx, fy = self._get_focal_px(cam_id)

        # Используем высоту бокса (более стабильна для грузовика)
        # Реальная высота кабины грузовика ~350 см
        H_real = self.truck_dims.height_cm

        # Оценка Z по вертикали
        z_from_height = (fy * H_real) / h_px

        # Оценка Z по ширине (грузовик шире, но aspect ratio может варьироваться)
        W_real = self.truck_dims.width_cm
        z_from_width = (fx * W_real) / w_px

        # Комбинируем (взвешенное среднее)
        # Высота обычно надежнее для вертикальных объектов
        z_cm = 0.7 * z_from_height + 0.3 * z_from_width

        # Ограничиваем разумными пределами
        z_cm = np.clip(z_cm, self.min_z_cm, self.max_z_cm)

        # Confidence на основе размера бокса
        # Чем больше бокс, тем ближе и точнее
        bbox_area = w_px * h_px
        img_area = self.camera_manager.get_camera(cam_id).intrinsics.width * \
                   self.camera_manager.get_camera(cam_id).intrinsics.height
        relative_area = bbox_area / img_area
        confidence = np.clip(relative_area * 50, 0.1, 0.9)

        # Временное сглаживание через Kalman Filter
        if track_id >= 0:
            kf = self._get_or_create_kf(cam_id, track_id)
            # Predict
            kf.predict()
            # Update
            measurement = np.array([[z_cm]], dtype=np.float32)
            kf.correct(measurement)
            z_smooth = kf.statePost[0, 0]
            z_cm = float(z_smooth)

            # История для дополнительного сглаживания
            self.z_history[(cam_id, track_id)].append(z_cm)
            if len(self.z_history[(cam_id, track_id)]) > self.max_history:
                self.z_history[(cam_id, track_id)].pop(0)

        return DepthEstimate(
            z_cm=z_cm,
            confidence=confidence,
            method="bbox_size",
            bbox_width_px=w_px,
            bbox_height_px=h_px
        )

    def estimate_from_temporal(
        self,
        cam_id: int,
        track_id: int
    ) -> Optional[DepthEstimate]:
        """Оценка глубины из временной истории (если текущий кадр не дал оценку)."""
        key = (cam_id, track_id)
        if key in self.z_history and len(self.z_history[key]) >= 2:
            # Экстраполируем по тренду
            z_vals = self.z_history[key]
            if len(z_vals) >= 3:
                # Линейная экстраполяция
                z_pred = 2 * z_vals[-1] - z_vals[-2]
                return DepthEstimate(
                    z_cm=np.clip(z_pred, self.min_z_cm, self.max_z_cm),
                    confidence=0.3,
                    method="temporal",
                    bbox_width_px=0,
                    bbox_height_px=0
                )
            else:
                return DepthEstimate(
                    z_cm=z_vals[-1],
                    confidence=0.4,
                    method="temporal",
                    bbox_width_px=0,
                    bbox_height_px=0
                )
        return None

    def estimate(
        self,
        cam_id: int,
        bbox: np.ndarray,
        track_id: int = -1
    ) -> Optional[DepthEstimate]:
        """
        Основной метод: фьюжн bbox + temporal оценок.
        """
        # 1. Оценка по размеру бокса
        est_bbox = self.estimate_from_bbox(cam_id, bbox, track_id)

        # 2. Оценка из истории (если есть)
        est_temp = self.estimate_from_temporal(cam_id, track_id) if track_id >= 0 else None

        if est_bbox is None and est_temp is None:
            return None

        if est_bbox is None:
            return est_temp

        if est_temp is None:
            return est_bbox

        # Фьюжн: взвешенное по confidence
        total_conf = est_bbox.confidence + est_temp.confidence
        if total_conf > 0:
            z_fused = (est_bbox.z_cm * est_bbox.confidence +
                       est_temp.z_cm * est_temp.confidence) / total_conf
            return DepthEstimate(
                z_cm=z_fused,
                confidence=min(total_conf, 1.0),
                method="fusion",
                bbox_width_px=est_bbox.bbox_width_px,
                bbox_height_px=est_bbox.bbox_height_px
            )

        return est_bbox

    def get_3d_point(
        self,
        cam_id: int,
        u: float,
        v: float,
        bbox: np.ndarray,
        track_id: int = -1,
        frame_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Получает полную 3D точку (X, Y, Z) из 2D центра бокса + моноскулрная глубина.

        Использует обратную проекцию: ray из центра камеры через точку (u,v) на глубине Z.
        """
        depth_est = self.estimate(cam_id, bbox, track_id)
        if depth_est is None:
            return None

        cam = self.camera_manager.get_camera(cam_id)
        z_cm = depth_est.z_cm

        # Обратная проекция через матрицу проекции
        # P = K [R | t], точка в мире: X = R^T (K^-1 * [u,v,1]^T * z - t)
        P = cam.get_projection_matrix()
        K = cam.intrinsics.K
        R = cam.extrinsics.R_cam_to_world
        t = cam.extrinsics.location

        # Нормализованные координаты камеры
        uv1 = np.array([u, v, 1.0], dtype=float)
        ray_cam = np.linalg.inv(K) @ uv1  # направление луча в системе камеры
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Точка в системе камеры на глубине z_cm
        point_cam = ray_cam * z_cm

        # Переводим в мировую систему
        point_world = R.T @ (point_cam - t)

        return point_world


class MultiCameraMonocularDepth:
    """
    Координирует моноскулрную глубину для всех камер.
    Используется когда триангуляция невозможна (1 камера).
    """

    def __init__(self, camera_manager, config: dict = None):
        self.estimator = MonocularDepthEstimator(camera_manager, config=config)
        self.camera_manager = camera_manager

    def process_single_camera_detections(
        self,
        detections_per_cam: Dict[int, List],
        frame_idx: int
    ) -> Dict[int, List[Tuple[int, np.ndarray, DepthEstimate]]]:
        """
        Обрабатывает детекции с камер, где < 2 детекций (нельзя триангулировать).
        Возвращает {cam_id: [(track_id, point_3d, depth_est), ...]}.
        """
        results = {}

        for cam_id, dets in detections_per_cam.items():
            if len(dets) == 0:
                continue

            # Для каждой детекции оцениваем глубину
            cam_results = []
            for det in dets:
                if det.track_id is not None:
                    point_3d = self.estimator.get_3d_point(
                        cam_id,
                        det.center[0],
                        det.center[1],
                        det.bbox,
                        track_id=det.track_id,
                        frame_idx=frame_idx
                    )
                    if point_3d is not None:
                        depth_est = self.estimator.estimate(
                            cam_id, det.bbox, det.track_id
                        )
                        cam_results.append((det.track_id, point_3d, depth_est))

            if cam_results:
                results[cam_id] = cam_results

        return results


def create_monocular_depth_estimator(
    camera_manager,
    config_path: str = None
) -> MultiCameraMonocularDepth:
    """Фабрика для создания моноскулрного оценщика глубины."""
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    mono_config = config.get('monocular_depth', {})
    return MultiCameraMonocularDepth(camera_manager, mono_config)