"""
Per-Camera Detector — детекция и 2D трекинг на каждой камере отдельно.
Использует YOLOv8 + BoTSORT/ByteTrack из Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import yaml

from ultralytics import YOLO


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Detection2D:
    """Результат детекции на одной камере."""
    cam_id: int
    frame_idx: int
    track_id: int                    # ID трека на этой камере (-1 если нет трекинга)
    class_id: int
    class_name: str
    bbox: np.ndarray                  # [x1, y1, x2, y2] в пикселях
    center: np.ndarray                # [cx, cy] центр бокса
    confidence: float
    timestamp: float = 0.0
    # Дополнительные поля для ReID
    embedding: Optional[np.ndarray] = None  # ReID эмбеддинг (512-d)
    mask: Optional[np.ndarray] = None       # Сегментационная маска (если есть)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @property
    def area(self) -> float:
        w, h = self.xywh[2], self.xywh[3]
        return w * h


@dataclass
class CameraDetections:
    """Все детекции на кадре для одной камеры."""
    cam_id: int
    frame_idx: int
    detections: List[Detection2D] = field(default_factory=list)
    timestamp: float = 0.0

    def get_by_track_id(self, track_id: int) -> Optional[Detection2D]:
        for det in self.detections:
            if det.track_id == track_id:
                return det
        return None

    def filter_by_class(self, class_ids: List[int]) -> 'CameraDetections':
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return CameraDetections(self.cam_id, self.frame_idx, filtered, self.timestamp)

    def filter_by_confidence(self, min_conf: float) -> 'CameraDetections':
        filtered = [d for d in self.detections if d.confidence >= min_conf]
        return CameraDetections(self.cam_id, self.frame_idx, filtered, self.timestamp)


# =============================================================================
# Per-Camera Tracker
# =============================================================================

class PerCameraTracker:
    """
    Обёртка над YOLO + BoTSORT/ByteTrack для одной камеры.
    Сохраняет состояния треков между кадрами.
    """

    def __init__(
        self,
        cam_id: int,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        tracker_type: str = "botsort",
        tracker_config: Optional[str] = None,
        classes: Optional[List[int]] = None,
        half: bool = False
    ):
        self.cam_id = cam_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.classes = classes

        # Загрузка модели
        print(f"[Cam{cam_id}] Загрузка модели: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        if half and device != "cpu":
            self.model.model.half()

        # Конфиг трекера
        self.tracker_type = tracker_type
        self.tracker_config = tracker_config

        # Кэш классов
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

        # Статистика
        self.frame_count = 0
        self.total_detections = 0

    def process(self, frame: np.ndarray, frame_idx: int, timestamp: float = 0.0) -> CameraDetections:
        """
        Обрабатывает один кадр, возвращает детекции с track_id.
        """
        self.frame_count += 1

        # Инференс с трекингом
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.classes,
            tracker=self.tracker_config if self.tracker_type else None,
            persist=True,           # Сохранять треки между кадрами
            verbose=False,
            stream=False
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                # Track IDs (если трекинг включен)
                track_ids = np.full(len(xyxy), -1, dtype=int)
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    class_id = clss[i]
                    class_name = self.class_names.get(class_id, str(class_id))

                    det = Detection2D(
                        cam_id=self.cam_id,
                        frame_idx=frame_idx,
                        track_id=int(track_ids[i]),
                        class_id=class_id,
                        class_name=class_name,
                        bbox=np.array([x1, y1, x2, y2], dtype=float),
                        center=np.array([cx, cy], dtype=float),
                        confidence=float(confs[i]),
                        timestamp=timestamp
                    )
                    detections.append(det)
                    self.total_detections += 1

        return CameraDetections(
            cam_id=self.cam_id,
            frame_idx=frame_idx,
            detections=detections,
            timestamp=timestamp
        )

    def reset(self):
        """Сброс состояния трекера."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cam_id": self.cam_id,
            "frames_processed": self.frame_count,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": self.total_detections / max(1, self.frame_count)
        }


# =============================================================================
# Multi-Camera Detector Manager
# =============================================================================

class MultiCameraDetector:
    """
    Менеджер детекторов для всех камер.
    Параллельно обрабатывает кадры со всех камер.
    """

    def __init__(self, camera_manager, config: Dict):
        self.camera_manager = camera_manager
        self.config = config
        self.detectors: Dict[int, PerCameraTracker] = {}
        self._init_detectors()

    def _init_detectors(self):
        det_config = self.config.get('detection', {})
        for cam_id in self.camera_manager.get_camera_ids():
            self.detectors[cam_id] = PerCameraTracker(
                cam_id=cam_id,
                model_path=det_config.get('model_path', 'models/yolov8n.pt'),
                conf_threshold=det_config.get('conf_threshold', 0.25),
                iou_threshold=det_config.get('iou_threshold', 0.45),
                imgsz=det_config.get('imgsz', 640),
                device=det_config.get('device', 'cpu'),
                tracker_type=det_config.get('tracker', 'botsort'),
                tracker_config=det_config.get('tracker_config'),
                classes=det_config.get('classes'),
                half=det_config.get('half', False)
            )
        print(f"Инициализировано детекторов: {len(self.detectors)}")

    def process_all(
        self,
        frames: Dict[int, np.ndarray],
        frame_idx: int,
        timestamp: float = 0.0
    ) -> Dict[int, CameraDetections]:
        """Обрабатывает кадры со всех камер."""
        results = {}
        for cam_id, frame in frames.items():
            if frame is not None and cam_id in self.detectors:
                results[cam_id] = self.detectors[cam_id].process(frame, frame_idx, timestamp)
            else:
                results[cam_id] = CameraDetections(cam_id=cam_id, frame_idx=frame_idx, timestamp=timestamp)
        return results

    def process_single(self, cam_id: int, frame: np.ndarray, frame_idx: int, timestamp: float = 0.0) -> CameraDetections:
        """Обрабатывает кадр одной камеры."""
        if cam_id in self.detectors and frame is not None:
            return self.detectors[cam_id].process(frame, frame_idx, timestamp)
        return CameraDetections(cam_id=cam_id, frame_idx=frame_idx, timestamp=timestamp)

    def get_all_stats(self) -> Dict[int, Dict]:
        return {cam_id: det.get_stats() for cam_id, det in self.detectors.items()}

    def reset_all(self):
        for det in self.detectors.values():
            det.reset()


# =============================================================================
# Factory
# =============================================================================

def create_multi_camera_detector(camera_manager, config_path: str = None) -> MultiCameraDetector:
    """Фабрика для создания MultiCameraDetector."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return MultiCameraDetector(camera_manager, config)


if __name__ == "__main__":
    # Тест
    from camera_manager import create_camera_manager

    cm = create_camera_manager()
    cm.open_all()

    detector = create_multi_camera_detector(cm)

    # Читаем первый кадр со всех камер
    frames = {}
    for cam_id in cm.get_camera_ids():
        ret, frame = cm.cameras[cam_id].read()
        if ret:
            frames[cam_id] = frame

    # Обрабатываем
    results = detector.process_all(frames, frame_idx=0)

    for cam_id, cam_dets in results.items():
        print(f"Cam{cam_id}: {len(cam_dets.detections)} детекций")
        for det in cam_dets.detections[:3]:
            print(f"  Track {det.track_id}: {det.class_name} @ {det.center}, conf={det.confidence:.3f}")

    cm.close_all()