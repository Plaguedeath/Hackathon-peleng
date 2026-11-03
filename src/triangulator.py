"""
3D Triangulator — триангуляция 3D точек из 2D наблюдений нескольких камер.
Поддерживает DLT, Optimal (Hartley-Sturm), и неопределённость через ковариации.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize

from .camera_manager import CameraManager, Camera
from .detector import Detection2D


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Observation2D:
    """Наблюдение 2D точки с одной камеры."""
    cam_id: int
    u: float
    v: float
    confidence: float = 1.0
    covariance: Optional[np.ndarray] = None  # 2x2 ковариация пикселя (если есть)
    track_id: int = -1
    frame_idx: int = 0


@dataclass
class TriangulationResult:
    """Результат триангуляции."""
    point_3d: np.ndarray                    # (3,) X, Y, Z в см
    covariance: np.ndarray                  # 3x3 ковариация 3D точки
    observations: List[Observation2D]       # Использованные наблюдения
    reprojection_errors: Dict[int, float]   # cam_id -> ошибка репроекции (px)
    success: bool = True
    method: str = "optimal"
    ransac_inliers: Optional[List[bool]] = None

    @property
    def mean_reproj_error(self) -> float:
        if not self.reprojection_errors:
            return float('inf')
        return np.mean(list(self.reprojection_errors.values()))

    @property
    def max_reproj_error(self) -> float:
        if not self.reprojection_errors:
            return float('inf')
        return np.max(list(self.reprojection_errors.values()))


# =============================================================================
# Triangulation Methods
# =============================================================================

def triangulate_dlt(observations: List[Observation2D], camera_manager: CameraManager,
                    frame: int = 0) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Direct Linear Transform (DLT) для триангуляции.
    Быстрый, линейный метод для инициализации.
    """
    if len(observations) < 2:
        raise ValueError("Need at least 2 observations for triangulation")

    A_rows = []
    cam_ids = []
    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        P = cam.get_projection_matrix(frame)
        # Уравнение: u * P3 - P1 = 0, v * P3 - P2 = 0
        A_rows.append(obs.u * P[2] - P[0])
        A_rows.append(obs.v * P[2] - P[1])
        cam_ids.extend([obs.cam_id, obs.cam_id])

    A = np.array(A_rows)
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    X = X_homogeneous[:3] / X_homogeneous[3]

    # Репроекционные ошибки
    reproj_errors = {}
    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        px = cam.project_world_to_image(X, frame)
        if px:
            reproj_errors[obs.cam_id] = np.sqrt((px[0] - obs.u)**2 + (px[1] - obs.v)**2)
        else:
            reproj_errors[obs.cam_id] = float('inf')

    return X, reproj_errors


def triangulate_optimal(observations: List[Observation2D], camera_manager: CameraManager,
                        frame: int = 0, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Оптимальная триангуляция (Hartley-Sturm) — минимизирует сумму квадратов репроекционных ошибок.
    """
    if len(observations) < 2:
        raise ValueError("Need at least 2 observations for triangulation")

    # Начальное приближение через DLT
    if initial_guess is None:
        X_init, _ = triangulate_dlt(observations, camera_manager, frame)
    else:
        X_init = initial_guess

    # Функция ошибки
    def reprojection_error(X):
        errors = []
        for obs in observations:
            cam = camera_manager.get_camera(obs.cam_id)
            px = cam.project_world_to_image(X, frame)
            if px:
                w = obs.confidence  # вес по confidence
                errors.append(w * (px[0] - obs.u)**2)
                errors.append(w * (px[1] - obs.v)**2)
        return np.sum(errors)

    # Оптимизация
    result = minimize(reprojection_error, X_init, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-9})

    X_opt = result.x

    # Репроекционные ошибки
    reproj_errors = {}
    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        px = cam.project_world_to_image(X_opt, frame)
        if px:
            reproj_errors[obs.cam_id] = np.sqrt((px[0] - obs.u)**2 + (px[1] - obs.v)**2)
        else:
            reproj_errors[obs.cam_id] = float('inf')

    return X_opt, reproj_errors


def triangulate_midpoint(observations: List[Observation2D], camera_manager: CameraManager,
                         frame: int = 0) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Midpoint triangulation — находит точку, минимизирующую сумму расстояний до лучей.
    Работает для 2+ камер.
    """
    if len(observations) < 2:
        raise ValueError("Need at least 2 observations")

    origins = []
    directions = []
    weights = []
    cam_ids = []

    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        origin, direction = cam.ray_from_pixel(obs.u, obs.v, frame)
        origins.append(origin)
        directions.append(direction / np.linalg.norm(direction))
        weights.append(obs.confidence)
        cam_ids.append(obs.cam_id)

    origins = np.array(origins)       # (N, 3)
    directions = np.array(directions) # (N, 3)
    weights = np.array(weights)       # (N,)

    # Для каждой пары лучей находим точку минимального расстояния
    # Затем усредняем с весами
    N = len(origins)
    points = []
    pair_weights = []

    for i in range(N):
        for j in range(i + 1, N):
            o1, d1 = origins[i], directions[i]
            o2, d2 = origins[j], directions[j]

            # Решаем: o1 + t1*d1 = o2 + t2*d2  (в смысле наименьших квадратов)
            # Минимизируем ||(o1 + t1*d1) - (o2 + t2*d2)||^2
            A = np.column_stack([d1, -d2])  # (3, 2)
            b = o2 - o1
            t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            midpoint = (o1 + t[0] * d1 + o2 + t[1] * d2) / 2
            points.append(midpoint)
            pair_weights.append(weights[i] * weights[j])

    if not points:
        raise ValueError("Could not compute midpoint")

    points = np.array(points)
    pair_weights = np.array(pair_weights)
    X = np.average(points, axis=0, weights=pair_weights)

    # Репроекционные ошибки
    reproj_errors = {}
    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        px = cam.project_world_to_image(X, frame)
        if px:
            reproj_errors[obs.cam_id] = np.sqrt((px[0] - obs.u)**2 + (px[1] - obs.v)**2)
        else:
            reproj_errors[obs.cam_id] = float('inf')

    return X, reproj_errors


def estimate_covariance(observations: List[Observation2D], camera_manager: CameraManager,
                        X: np.ndarray, frame: int = 0) -> np.ndarray:
    """
    Оценка ковариации 3D точки через распространение неопределённости (error propagation).
    Использует якобиан проекции.
    """
    if len(observations) < 2:
        return np.eye(3) * 1e6  # Большая неопределённость

    J_rows = []
    W_rows = []

    for obs in observations:
        cam = camera_manager.get_camera(obs.cam_id)
        # Якобиан проекции d(u,v)/d(X,Y,Z)
        J = _projection_jacobian(cam, X, frame)
        J_rows.append(J)

        # Ковариация измерения (2x2)
        if obs.covariance is not None:
            W_obs = np.linalg.inv(obs.covariance)
        else:
            # Приблизительная ковариация из confidence
            sigma_pix = max(1.0, 2.0 * (1.0 - obs.confidence))  # чем выше conf, тем меньше sigma
            W_obs = np.eye(2) / (sigma_pix ** 2)
        W_rows.append(W_obs)

    J = np.vstack(J_rows)           # (2N, 3)
    W = np.block([[W_rows[i] if i == j else np.zeros((2, 2))
                   for j in range(len(W_rows))] for i in range(len(W_rows))])

    # Ковариация 3D: (J^T W J)^-1
    try:
        cov = np.linalg.inv(J.T @ W @ J)
    except np.linalg.LinAlgError:
        cov = np.eye(3) * 1e6

    return cov


def _projection_jacobian(cam: Camera, X: np.ndarray, frame: int) -> np.ndarray:
    """Вычисляет якобиан проекции 3D -> 2D: d(u,v)/d(X,Y,Z)."""
    R = cam.extrinsics.get_rotation(frame)
    forward = R[:, 0].copy()
    right = R[:, 1].copy()
    down = np.cross(right, forward)
    down = down / np.linalg.norm(down)
    W2C = np.stack([right, down, forward], axis=0)
    t = cam.extrinsics.location.reshape(3, 1)

    X_world = X.reshape(3, 1)
    X_cam = (W2C @ (X_world - t)).flatten()

    Xc, Yc, Zc = X_cam[0], X_cam[1], X_cam[2]
    fx, fy = cam.intrinsics.fx, cam.intrinsics.fy

    # d(u,v)/d(Xc,Yc,Zc)
    du_dXc = fx / Zc
    du_dYc = 0
    du_dZc = -fx * Xc / (Zc ** 2)

    dv_dXc = 0
    dv_dYc = fy / Zc
    dv_dZc = -fy * Yc / (Zc ** 2)

    J_cam = np.array([
        [du_dXc, du_dYc, du_dZc],
        [dv_dXc, dv_dYc, dv_dZc]
    ])

    # d(Xc,Yc,Zc)/d(X,Y,Z) = W2C
    J = J_cam @ W2C
    return J


# =============================================================================
# RANSAC для устойчивости к выбросам
# =============================================================================

def triangulate_ransac(observations: List[Observation2D], camera_manager: CameraManager,
                       frame: int = 0, iterations: int = 100,
                       threshold_px: float = 5.0, min_inliers: int = 2,
                       method: str = "optimal") -> Tuple[np.ndarray, Dict[int, float], List[bool]]:
    """
    RANSAC триангуляция для отбраковки выбросов в наблюдениях.
    Возвращает лучшую модель, ошибки и маску инлаеров.
    """
    if len(observations) < min_inliers:
        raise ValueError(f"Need at least {min_inliers} observations")

    best_X = None
    best_errors = {}
    best_inliers = None
    best_score = -1

    obs_list = list(observations)
    n_obs = len(obs_list)

    for _ in range(iterations):
        # Случайная выборка минимального набора (2 камеры)
        if n_obs > 2:
            idx = np.random.choice(n_obs, size=2, replace=False)
            sample = [obs_list[i] for i in idx]
        else:
            sample = obs_list

        try:
            if method == "optimal":
                X, errors = triangulate_optimal(sample, camera_manager, frame)
            elif method == "midpoint":
                X, errors = triangulate_midpoint(sample, camera_manager, frame)
            else:
                X, errors = triangulate_dlt(sample, camera_manager, frame)
        except Exception:
            continue

        # Оценка всех наблюдений
        inliers = []
        for i, obs in enumerate(obs_list):
            cam = camera_manager.get_camera(obs.cam_id)
            px = cam.project_world_to_image(X, frame)
            if px:
                err = np.sqrt((px[0] - obs.u)**2 + (px[1] - obs.v)**2)
                inliers.append(err < threshold_px)
            else:
                inliers.append(False)

        n_inliers = sum(inliers)
        if n_inliers > best_score:
            best_score = n_inliers
            best_X = X
            best_errors = errors
            best_inliers = inliers

    if best_X is None or best_score < min_inliers:
        # Fallback: обычная триангуляция по всем
        if method == "optimal":
            X, errors = triangulate_optimal(observations, camera_manager, frame)
        elif method == "midpoint":
            X, errors = triangulate_midpoint(observations, camera_manager, frame)
        else:
            X, errors = triangulate_dlt(observations, camera_manager, frame)
        return X, errors, [True] * len(observations)

    # Рефайним по всем инлайерам
    inlier_obs = [obs for obs, inl in zip(observations, best_inliers) if inl]
    if len(inlier_obs) >= 2:
        if method == "optimal":
            X, errors = triangulate_optimal(inlier_obs, camera_manager, frame, initial_guess=best_X)
        elif method == "midpoint":
            X, errors = triangulate_midpoint(inlier_obs, camera_manager, frame)
        else:
            X, errors = triangulate_dlt(inlier_obs, camera_manager, frame)

    return X, errors, best_inliers


# =============================================================================
# Main Triangulator Class
# =============================================================================

class Triangulator:
    """
    Основной класс для триангуляции с поддержкой разных методов,
    RANSAC, ковариаций и фильтрации.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        method: str = "optimal",
        min_cameras: int = 2,
        max_reproj_error_px: float = 20.0,
        ransac_iterations: int = 100,
        ransac_threshold_px: float = 5.0,
        use_uncertainty: bool = True
    ):
        self.camera_manager = camera_manager
        self.method = method
        self.min_cameras = min_cameras
        self.max_reproj_error_px = max_reproj_error_px
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold_px = ransac_threshold_px
        self.use_uncertainty = use_uncertainty

        self.stats = {
            'total_calls': 0,
            'successful': 0,
            'failed_insufficient': 0,
            'failed_reproj': 0,
            'failed_ransac': 0
        }

    def triangulate(
        self,
        detections_per_cam: Dict[int, List[Detection2D]],
        frame_idx: int,
        use_ransac: bool = True
    ) -> List[TriangulationResult]:
        """
        Триангулирует 3D точки из детекций на текущем кадре.
        Группирует детекции по track_id внутри камеры, затем ассоциирует между камерами.
        """
        self.stats['total_calls'] += 1

        # Собираем наблюдения по track_id для каждой камеры
        obs_per_cam = {}
        for cam_id, dets in detections_per_cam.items():
            obs_per_cam[cam_id] = {}
            for det in dets:
                if det.track_id >= 0:  # Только если есть track_id
                    obs = Observation2D(
                        cam_id=cam_id,
                        u=det.center[0],
                        v=det.center[1],
                        confidence=det.confidence,
                        track_id=det.track_id,
                        frame_idx=frame_idx
                    )
                    obs_per_cam[cam_id][det.track_id] = obs

        # Находим track_id, которые видны в >= min_cameras камерах
        all_track_ids = set()
        for cam_obs in obs_per_cam.values():
            all_track_ids.update(cam_obs.keys())

        results = []
        for track_id in all_track_ids:
            # Собираем наблюдения этого track_id со всех камер
            observations = []
            for cam_id in obs_per_cam:
                if track_id in obs_per_cam[cam_id]:
                    observations.append(obs_per_cam[cam_id][track_id])

            if len(observations) < self.min_cameras:
                self.stats['failed_insufficient'] += 1
                continue

            # Триангуляция
            try:
                if use_ransac and self.ransac_iterations > 0 and len(observations) > 2:
                    X, reproj_errors, inliers = triangulate_ransac(
                        observations, self.camera_manager, frame_idx,
                        self.ransac_iterations, self.ransac_threshold_px,
                        self.min_cameras, self.method
                    )
                    if sum(inliers) < self.min_cameras:
                        self.stats['failed_ransac'] += 1
                        continue
                    # Фильтруем наблюдения по инлайерам
                    observations = [obs for obs, inl in zip(observations, inliers) if inl]
                else:
                    if self.method == "optimal":
                        X, reproj_errors = triangulate_optimal(observations, self.camera_manager, frame_idx)
                    elif self.method == "midpoint":
                        X, reproj_errors = triangulate_midpoint(observations, self.camera_manager, frame_idx)
                    else:
                        X, reproj_errors = triangulate_dlt(observations, self.camera_manager, frame_idx)

                # Проверка репроекционной ошибки
                max_err = max(reproj_errors.values()) if reproj_errors else float('inf')
                if max_err > self.max_reproj_error_px:
                    self.stats['failed_reproj'] += 1
                    continue

                # Ковариация
                covariance = np.eye(3) * 1e-6
                if self.use_uncertainty:
                    covariance = estimate_covariance(observations, self.camera_manager, X, frame_idx)

                result = TriangulationResult(
                    point_3d=X,
                    covariance=covariance,
                    observations=observations,
                    reprojection_errors=reproj_errors,
                    success=True,
                    method=self.method,
                    ransac_inliers=inliers if use_ransac and self.ransac_iterations > 0 else None
                )
                results.append(result)
                self.stats['successful'] += 1

            except Exception as e:
                print(f"[Triangulator] Ошибка триангуляции track {track_id}: {e}")
                continue

        return results

    def triangulate_single(
        self,
        observations: List[Observation2D],
        frame_idx: int,
        use_ransac: bool = True
    ) -> Optional[TriangulationResult]:
        """Триангулирует одну 3D точку из списка наблюдений."""
        if len(observations) < self.min_cameras:
            return None

        try:
            if use_ransac and self.ransac_iterations > 0 and len(observations) > 2:
                X, reproj_errors, inliers = triangulate_ransac(
                    observations, self.camera_manager, frame_idx,
                    self.ransac_iterations, self.ransac_threshold_px,
                    self.min_cameras, self.method
                )
                if sum(inliers) < self.min_cameras:
                    return None
                observations = [obs for obs, inl in zip(observations, inliers) if inl]
            else:
                if self.method == "optimal":
                    X, reproj_errors = triangulate_optimal(observations, self.camera_manager, frame_idx)
                elif self.method == "midpoint":
                    X, reproj_errors = triangulate_midpoint(observations, self.camera_manager, frame_idx)
                else:
                    X, reproj_errors = triangulate_dlt(observations, self.camera_manager, frame_idx)

            max_err = max(reproj_errors.values()) if reproj_errors else float('inf')
            if max_err > self.max_reproj_error_px:
                return None

            covariance = np.eye(3) * 1e-6
            if self.use_uncertainty:
                covariance = estimate_covariance(observations, self.camera_manager, X, frame_idx)

            return TriangulationResult(
                point_3d=X,
                covariance=covariance,
                observations=observations,
                reprojection_errors=reproj_errors,
                success=True,
                method=self.method,
                ransac_inliers=inliers if use_ransac else None
            )

        except Exception as e:
            print(f"[Triangulator] Ошибка: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0


# =============================================================================
# Factory
# =============================================================================

def create_triangulator(camera_manager: CameraManager, config_path: str = None) -> Triangulator:
    """Фабрика для создания Triangulator."""
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tri_config = config.get('triangulation', {})
    return Triangulator(
        camera_manager=camera_manager,
        method=tri_config.get('method', 'optimal'),
        min_cameras=tri_config.get('min_cameras', 2),
        max_reproj_error_px=tri_config.get('max_reproj_error_px', 20.0),
        ransac_iterations=tri_config.get('ransac_iterations', 100),
        ransac_threshold_px=tri_config.get('ransac_threshold_px', 5.0),
        use_uncertainty=tri_config.get('use_uncertainty', True)
    )


if __name__ == "__main__":
    # Тест
    from camera_manager import create_camera_manager
    from detector import create_multi_camera_detector

    cm = create_camera_manager()
    cm.open_all()
    detector = create_multi_camera_detector(cm)
    triangulator = create_triangulator(cm)

    # Читаем первый кадр
    frames = {}
    for cam_id in cm.get_camera_ids():
        ret, frame = cm.cameras[cam_id].read()
        if ret:
            frames[cam_id] = frame

    # Детекция
    dets = detector.process_all(frames, frame_idx=0)
    print("Детекции:", {k: len(v.detections) for k, v in dets.items()})

    # Триангуляция
    results = triangulator.triangulate(dets, frame_idx=0)
    print(f"Триангулировано точек: {len(results)}")
    for r in results:
        print(f"  Track IDs: {[o.track_id for o in r.observations]}, "
              f"3D: {r.point_3d}, reproj: {r.mean_reproj_error:.2f}px")

    cm.close_all()