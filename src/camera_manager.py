"""
Camera Manager — управление множественными калиброванными камерами.
Загружает конфигурацию, строит матрицы проекции, предоставляет методы
проекции 3D↔2D и построения лучей для триангуляции.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Intrinsics:
    """Внутренние параметры камеры."""
    K: np.ndarray                    # 3x3 матрица камеры
    width: int
    height: int
    focal_mm: float
    sensor_width_mm: float
    sensor_height_mm: float

    @property
    def fx(self) -> float: return self.K[0, 0]
    @property
    def fy(self) -> float: return self.K[1, 1]
    @property
    def cx(self) -> float: return self.K[0, 2]
    @property
    def cy(self) -> float: return self.K[1, 2]

    def project(self, X_cam: np.ndarray) -> Optional[Tuple[float, float]]:
        """Проецирует точку в системе камеры (X, Y, Z) -> (u, v)."""
        if X_cam[2] <= 0:
            return None
        u = self.fx * X_cam[0] / X_cam[2] + self.cx
        v = self.fy * X_cam[1] / X_cam[2] + self.cy
        if 0 <= u < self.width and 0 <= v < self.height:
            return (float(u), float(v))
        return None


@dataclass
class Extrinsics:
    """Внешние параметры камеры (поза в мире)."""
    location: np.ndarray             # (3,) положение камеры в мире (см)
    rotation_deg: np.ndarray         # (3,) roll, pitch, yaw в градусах (UE)
    R_world_to_cam: np.ndarray       # 3x3 матрица поворота мир -> камера
    R_cam_to_world: np.ndarray       # 3x3 матрица поворота камера -> мир
    is_static: bool = True
    yaw_per_frame: Optional[Dict[int, float]] = None  # для вращающихся камер

    def get_rotation(self, frame: int) -> np.ndarray:
        """Возвращает матрицу поворота для конкретного кадра."""
        if self.yaw_per_frame and frame in self.yaw_per_frame:
            # Обновляем только yaw
            roll, pitch, _ = self.rotation_deg
            yaw = self.yaw_per_frame[frame]
            return self._build_rotation_matrix(roll, pitch, yaw)
        return self.R_world_to_cam

    @staticmethod
    def _build_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """Строит матрицу поворота R = Rz(yaw) @ Ry(pitch) @ Rx(roll) для UE.

        Важно: UE конвенция Roll наоборот — положительный Roll = правое крыло ВНИЗ.
        Наша Rx матрица поворачивает +Y к +Z (правое крыло ВВЕРХ) при положительном угле.
        Поэтому Roll нужно инвертировать: roll_ue = -roll_csv.
        """
        r, p, y = map(math.radians, (-roll_deg, pitch_deg, yaw_deg))
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr,  cr]], dtype=float)
        Ry = np.array([[cp,  0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]], dtype=float)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)
        return Rz @ Ry @ Rx


@dataclass
class Camera:
    """Полная информация о камере."""
    id: int
    name: str
    source: str
    intrinsics: Intrinsics
    extrinsics: Extrinsics
    cap: Optional[cv2.VideoCapture] = field(default=None, repr=False)
    fps: float = 30.0
    total_frames: int = 0
    current_frame: int = 0

    def open(self) -> bool:
        """Открывает видеопоток."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Читает следующий кадр."""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def seek(self, frame_idx: int) -> bool:
        """Переходит к конкретному кадру."""
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame = frame_idx
        return True

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def project_world_to_image(self, point_world: np.ndarray, frame: int = 0) -> Optional[Tuple[float, float]]:
        """Проецирует мировую точку (UE, см) на пиксель (u, v)."""
        R = self.extrinsics.get_rotation(frame)
        # UE конвенция: forward = столбец 0, right = столбец 1, down = cross(right, forward)
        forward = R[:, 0].copy()
        right = R[:, 1].copy()
        down = np.cross(right, forward)
        down = down / np.linalg.norm(down)
        # Матрица мир -> камера: строки [right, down, forward]
        W2C = np.stack([right, down, forward], axis=0)
        t = self.extrinsics.location.reshape(3, 1)

        point_world = np.asarray(point_world, dtype=float).reshape(3, 1)
        X_cam = (W2C @ (point_world - t)).flatten()
        return self.intrinsics.project(X_cam)

    def ray_from_pixel(self, u: float, v: float, frame: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (origin, direction) луча в мировых координатах для пикселя (u, v)."""
        R = self.extrinsics.get_rotation(frame)
        forward = R[:, 0].copy()
        right = R[:, 1].copy()
        down = np.cross(right, forward)
        down = down / np.linalg.norm(down)
        W2C = np.stack([right, down, forward], axis=0)

        # Пиксель -> луч в системе камеры
        kinv = np.linalg.inv(self.intrinsics.K)
        pixel = np.array([u, v, 1.0], dtype=float)
        ray_cam = kinv @ pixel
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Камера -> мир
        ray_world = W2C.T @ ray_cam
        origin = self.extrinsics.location.copy()
        return origin, ray_world

    def get_projection_matrix(self, frame: int = 0) -> np.ndarray:
        """Возвращает матрицу проекции P = K @ [R | t] (3x4)."""
        R = self.extrinsics.get_rotation(frame)
        forward = R[:, 0].copy()
        right = R[:, 1].copy()
        down = np.cross(right, forward)
        down = down / np.linalg.norm(down)
        W2C = np.stack([right, down, forward], axis=0)
        t = self.extrinsics.location.reshape(3, 1)
        Rt = np.hstack([W2C, -W2C @ t])
        return self.intrinsics.K @ Rt


class CameraManager:
    """Менеджер множественных камер."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.cameras: Dict[int, Camera] = {}
        self._load_config()
        self._validate_calibration()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.coordinate_system = config.get('coordinate_system', 'UE')
        self.world_frame = config.get('world_frame', 'camera_1')

        for cam_config in config['cameras']:
            cam_id = cam_config['id']

            # Intrinsics
            intr = cam_config['intrinsics']
            fx = intr['focal_mm'] * intr['image_width'] / intr['sensor_width_mm']
            fy = intr['focal_mm'] * intr['image_height'] / intr['sensor_height_mm']
            cx = intr['image_width'] / 2.0
            cy = intr['image_height'] / 2.0
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=float)
            intrinsics = Intrinsics(
                K=K,
                width=intr['image_width'],
                height=intr['image_height'],
                focal_mm=intr['focal_mm'],
                sensor_width_mm=intr['sensor_width_mm'],
                sensor_height_mm=intr['sensor_height_mm']
            )

            # Extrinsics
            extr = cam_config['extrinsics']
            location = np.array(extr['location_cm'], dtype=float)
            rotation = np.array(extr['rotation_deg'], dtype=float)

            # Базовая матрица поворота
            R_base = Extrinsics._build_rotation_matrix(rotation[0], rotation[1], rotation[2])
            forward = R_base[:, 0].copy()
            right = R_base[:, 1].copy()
            down = np.cross(right, forward)
            down = down / np.linalg.norm(down)
            W2C = np.stack([right, down, forward], axis=0)

            # Загрузка yaw_per_frame если есть
            yaw_per_frame = None
            if not extr.get('is_static', True) and 'yaw_per_frame_csv' in extr:
                yaw_per_frame = self._load_yaw_csv(extr['yaw_per_frame_csv'])

            extrinsics = Extrinsics(
                location=location,
                rotation_deg=rotation,
                R_world_to_cam=W2C,
                R_cam_to_world=W2C.T,
                is_static=extr.get('is_static', True),
                yaw_per_frame=yaw_per_frame
            )

            camera = Camera(
                id=cam_id,
                name=cam_config['name'],
                source=cam_config['source'],
                intrinsics=intrinsics,
                extrinsics=extrinsics
            )
            self.cameras[cam_id] = camera

        print(f"Загружено камер: {len(self.cameras)}")
        for cam in self.cameras.values():
            print(f"  Cam{cam.id} ({cam.name}): {cam.intrinsics.width}x{cam.intrinsics.height}, "
                  f"f={cam.intrinsics.focal_mm}mm, static={cam.extrinsics.is_static}")

    def _load_yaw_csv(self, csv_path: str) -> Dict[int, float]:
        """Загружает yaw для каждого кадра из CSV камеры 5."""
        yaw_dict = {}
        try:
            import csv
            with open(csv_path, 'r', encoding='cp1251', errors='replace') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 7:
                        frame = int(row[0])
                        yaw = float(row[6].replace(',', '.'))
                        yaw_dict[frame] = yaw
            print(f"  Загружены yaw для {len(yaw_dict)} кадров из {csv_path}")
        except Exception as e:
            print(f"  Предупреждение: не удалось загрузить yaw из {csv_path}: {e}")
        return yaw_dict

    def _validate_calibration(self):
        """Проверка калибровки: репроекция известных точек."""
        pass

    def open_all(self) -> bool:
        """Открывает все видеопотоки."""
        success = True
        for cam in self.cameras.values():
            if not cam.open():
                print(f"Ошибка открытия камеры {cam.id}: {cam.source}")
                success = False
        return success

    def read_all(self) -> Dict[int, Optional[np.ndarray]]:
        """Синхронно читает кадр со всех камер."""
        results = {}
        for cam_id, cam in self.cameras.items():
            ret, frame = cam.read()
            results[cam_id] = frame if ret else None
        return results

    def seek_all(self, frame_idx: int):
        """Переходит к кадру на всех камерах."""
        for cam in self.cameras.values():
            cam.seek(frame_idx)

    def close_all(self):
        for cam in self.cameras.values():
            cam.close()

    def get_camera(self, cam_id: int) -> Optional[Camera]:
        return self.cameras.get(cam_id)

    def get_camera_ids(self) -> List[int]:
        return sorted(self.cameras.keys())

    def project_world_to_image(self, cam_id: int, point_world: np.ndarray, frame: int = 0) -> Optional[Tuple[float, float]]:
        cam = self.cameras.get(cam_id)
        if cam is None:
            return None
        return cam.project_world_to_image(point_world, frame)

    def ray_from_pixel(self, cam_id: int, u: float, v: float, frame: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        cam = self.cameras[cam_id]
        return cam.ray_from_pixel(u, v, frame)

    def get_projection_matrix(self, cam_id: int, frame: int = 0) -> np.ndarray:
        cam = self.cameras[cam_id]
        return cam.get_projection_matrix(frame)


# =============================================================================
# Convenience functions
# =============================================================================

def create_camera_manager(config_path: Union[str, Path] = None) -> CameraManager:
    """Фабрика для создания CameraManager."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "cameras.json"
    return CameraManager(config_path)


if __name__ == "__main__":
    # Тест
    cm = create_camera_manager()
    cm.open_all()

    # Тест проекции для камеры 1
    test_point = np.array([-21500.0, -12800.0, 300.0])
    for cam_id in [1, 3, 5]:
        px = cm.project_world_to_image(cam_id, test_point, frame=0)
        print(f"Cam{cam_id}: {test_point} -> {px}")

    cm.close_all()