"""
Multi-Camera Data Association — связывание детекций между камерами.
Поддерживает геометрический (эпиполярный), appearance (ReID) и гибридный методы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .camera_manager import CameraManager
from .detector import Detection2D, CameraDetections
from .triangulator import Observation2D, TriangulationResult, create_triangulator


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MatchedGroup:
    """Группа сопоставленных детекций одного объекта на разных камерах."""
    track_id_3d: int = -1
    detections: Dict[int, Detection2D] = field(default_factory=dict)  # cam_id -> Detection2D
    triangulation: Optional[TriangulationResult] = None
    confidence: float = 0.0


# =============================================================================
# ReID Extractor (optional)
# =============================================================================

class ReIDExtractor:
    """
    Экстрактор appearance эмбеддингов для ReID.
    Использует OSNet или другие модели.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if self.model_path and Path(self.model_path).exists():
            try:
                import torch
                import torchreid
                self.model = torchreid.models.build_model(
                    name='osnet_x1_0',
                    num_classes=1000,
                    pretrained=False
                )
                torchreid.utils.load_pretrained_weights(self.model, self.model_path)
                self.model.to(self.device)
                self.model.eval()
                print(f"[ReID] Загружена модель: {self.model_path}")
            except Exception as e:
                print(f"[ReID] Ошибка загрузки модели: {e}")
                self.model = None
        else:
            print("[ReID] Модель не найдена, ReID отключён")

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Извлекает эмбеддинг из bbox на кадре."""
        if self.model is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Preprocess
        crop = cv2.resize(crop, (128, 256))
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        crop = crop.transpose(2, 0, 1)  # HWC -> CHW
        crop = np.expand_dims(crop, 0)

        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(crop).to(self.device)
            feat = self.model(tensor)
            feat = feat.cpu().numpy().flatten()
            feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat


# =============================================================================
# Geometric Association
# =============================================================================

class GeometricAssociator:
    """
    Геометрическая ассоциация через эпиполярную геометрию и триангуляцию.
    """

    def __init__(self, camera_manager: CameraManager, triangulator, max_epipolar_dist: float = 10.0):
        self.camera_manager = camera_manager
        self.triangulator = triangulator
        self.max_epipolar_dist = max_epipolar_dist
        self._fundamental_matrices = {}
        self._compute_fundamental_matrices()

    def _compute_fundamental_matrices(self):
        """Предвычисляет фундаментальные матрицы для всех пар камер."""
        cam_ids = self.camera_manager.get_camera_ids()
        for i, cam_id1 in enumerate(cam_ids):
            for cam_id2 in cam_ids[i+1:]:
                F = self._compute_fundamental_matrix(cam_id1, cam_id2)
                if F is not None:
                    self._fundamental_matrices[(cam_id1, cam_id2)] = F
                    self._fundamental_matrices[(cam_id2, cam_id1)] = F.T
        print(f"[GeometricAssociator] Вычислено F-матриц: {len(self._fundamental_matrices)//2}")

    def _compute_fundamental_matrix(self, cam_id1: int, cam_id2: int) -> Optional[np.ndarray]:
        """Вычисляет фундаментальную матрицу F между двумя камерами."""
        cam1 = self.camera_manager.get_camera(cam_id1)
        cam2 = self.camera_manager.get_camera(cam_id2)
        if cam1 is None or cam2 is None:
            return None

        # P1, P2 - матрицы проекции (3x4)
        P1 = cam1.get_projection_matrix()
        P2 = cam2.get_projection_matrix()

        # Essential matrix: E = [t]_x * R
        # где R, t - относительная поза cam2 -> cam1
        R1 = cam1.extrinsics.R_cam_to_world
        R2 = cam2.extrinsics.R_cam_to_world
        t1 = cam1.extrinsics.location
        t2 = cam2.extrinsics.location

        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1

        # Skew-symmetric matrix
        tx = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0]
        ])
        E = tx @ R_rel

        # Fundamental matrix: F = K2^-T * E * K1^-1
        K1 = cam1.intrinsics.K
        K2 = cam2.intrinsics.K
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        return F

    def epipolar_distance(self, cam_id1: int, u1: float, v1: float,
                          cam_id2: int, u2: float, v2: float) -> float:
        """Вычисляет расстояние от точки (u2,v2) до эпиполярной линии точки (u1,v1)."""
        key = (cam_id1, cam_id2)
        if key not in self._fundamental_matrices:
            return float('inf')

        F = self._fundamental_matrices[key]
        x1 = np.array([u1, v1, 1.0])
        x2 = np.array([u2, v2, 1.0])

        # Эпиполярная линия во 2-й камере: l2 = F * x1
        l2 = F @ x1
        # Расстояние от точки до линии: |l2^T * x2| / ||l2[:2]||
        dist = abs(l2 @ x2) / (np.linalg.norm(l2[:2]) + 1e-6)
        return float(dist)

    def geometric_cost(self, det1: Detection2D, det2: Detection2D) -> float:
        """Стоимость ассоциации двух детекций (геометрическая)."""
        return self.epipolar_distance(det1.cam_id, det1.center[0], det1.center[1],
                                      det2.cam_id, det2.center[0], det2.center[1])

    def build_cost_matrix(self, detections_per_cam: Dict[int, List[Detection2D]],
                          cam_order: List[int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Строит матрицу стоимости для ассоциации детекций между камерами.
        Возвращает (cost_matrix, index_mapping) где index_mapping[i] = (cam_id, local_idx)
        """
        if len(cam_order) < 2:
            return np.empty((0, 0)), []

        # Собираем все детекции в плоский список
        all_dets = []
        for cam_id in cam_order:
            dets = detections_per_cam.get(cam_id, [])
            for local_idx, det in enumerate(dets):
                all_dets.append((cam_id, local_idx, det))

        n = len(all_dets)
        cost = np.full((n, n), 1e6)
        np.fill_diagonal(cost, 0)

        # Заполняем попарные стоимости
        for i in range(n):
            cam_i, _, det_i = all_dets[i]
            for j in range(i+1, n):
                cam_j, _, det_j = all_dets[j]
                if cam_i == cam_j:
                    continue  # Не ассоциируем внутри одной камеры
                cost[i, j] = self.geometric_cost(det_i, det_j)
                cost[j, i] = cost[i, j]

        return cost, all_dets

    def associate_greedy(self, detections_per_cam: Dict[int, List[Detection2D]],
                         cam_order: List[int], threshold: float = None) -> List[MatchedGroup]:
        """Жадная ассоциация: строим кластеры детекций с малыми попарными стоимостями."""
        if threshold is None:
            threshold = self.max_epipolar_dist

        cost, all_dets = self.build_cost_matrix(detections_per_cam, cam_order)
        if cost.size == 0:
            return []

        n = len(all_dets)
        used = np.zeros(n, dtype=bool)
        groups = []

        for i in range(n):
            if used[i]:
                continue
            cam_i, local_i, det_i = all_dets[i]

            # Начинаем новую группу
            group_dets = {cam_i: det_i}
            used[i] = True

            # Ищем соседей
            for j in range(i+1, n):
                if used[j]:
                    continue
                cam_j, local_j, det_j = all_dets[j]
                if cam_j in group_dets:
                    continue  # Уже есть детекция с этой камеры

                if cost[i, j] < threshold:
                    group_dets[cam_j] = det_j
                    used[j] = True

            if len(group_dets) >= 2:
                # Триангуляция для верификации
                obs_list = []
                for cam_id, det in group_dets.items():
                    obs_list.append(Observation2D(
                        cam_id=cam_id,
                        u=det.center[0],
                        v=det.center[1],
                        confidence=det.confidence,
                        track_id=det.track_id,
                        frame_idx=det.frame_idx
                    ))

                tri_result = self.triangulator.triangulate_single(obs_list, det_i.frame_idx, use_ransac=False)
                if tri_result and tri_result.success:
                    groups.append(MatchedGroup(
                        detections=group_dets,
                        triangulation=tri_result,
                        confidence=np.mean([d.confidence for d in group_dets.values()])
                    ))

        return groups


# =============================================================================
# Appearance Association
# =============================================================================

class AppearanceAssociator:
    """
    Ассоциация по appearance (ReID эмбеддинги).
    """

    def __init__(self, reid_extractor: Optional[ReIDExtractor] = None,
                 threshold: float = 0.6):
        self.reid = reid_extractor
        self.threshold = threshold
        self.gallery: Dict[int, np.ndarray] = {}  # global_id -> embedding

    def extract_embeddings(self, frames: Dict[int, np.ndarray],
                           detections_per_cam: Dict[int, List[Detection2D]]) -> Dict[int, List[Detection2D]]:
        """Извлекает эмбеддинги для всех детекций."""
        if self.reid is None:
            return detections_per_cam

        for cam_id, dets in detections_per_cam.items():
            frame = frames.get(cam_id)
            if frame is None:
                continue
            for det in dets:
                if det.embedding is None:
                    det.embedding = self.reid.extract(frame, det.bbox)
        return detections_per_cam

    def appearance_cost(self, det1: Detection2D, det2: Detection2D) -> float:
        """Косинусное расстояние между эмбеддингами."""
        if det1.embedding is None or det2.embedding is None:
            return 1.0  # Максимальная стоимость если нет эмбеддингов
        sim = np.dot(det1.embedding, det2.embedding)
        return 1.0 - sim  # 0 = одинаковые, 1 = разные

    def associate(self, detections_per_cam: Dict[int, List[Detection2D]],
                  cam_order: List[int]) -> List[MatchedGroup]:
        """Ассоциация только по appearance (для камер без геометрии)."""
        if self.reid is None:
            return []

        # Аналогично geometric но с appearance cost
        all_dets = []
        for cam_id in cam_order:
            dets = detections_per_cam.get(cam_id, [])
            for local_idx, det in enumerate(dets):
                all_dets.append((cam_id, local_idx, det))

        n = len(all_dets)
        if n < 2:
            return []

        cost = np.full((n, n), 1.0)
        for i in range(n):
            for j in range(i+1, n):
                cam_i, _, det_i = all_dets[i]
                cam_j, _, det_j = all_dets[j]
                if cam_i != cam_j:
                    cost[i, j] = self.appearance_cost(det_i, det_j)
                    cost[j, i] = cost[i, j]

        # Greedy clustering
        used = np.zeros(n, dtype=bool)
        groups = []

        for i in range(n):
            if used[i]:
                continue
            cam_i, local_i, det_i = all_dets[i]
            group_dets = {cam_i: det_i}
            used[i] = True

            for j in range(i+1, n):
                if used[j]:
                    continue
                cam_j, local_j, det_j = all_dets[j]
                if cam_j in group_dets:
                    continue
                if cost[i, j] < (1.0 - self.threshold):
                    group_dets[cam_j] = det_j
                    used[j] = True

            if len(group_dets) >= 2:
                groups.append(MatchedGroup(
                    detections=group_dets,
                    confidence=np.mean([d.confidence for d in group_dets.values()])
                ))

        return groups


# =============================================================================
# Hybrid Associator
# =============================================================================

class HybridAssociator:
    """
    Гибридная ассоциация: геометрия + appearance + временная согласованность.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        triangulator,
        reid_extractor: Optional[ReIDExtractor] = None,
        geometric_weight: float = 0.6,
        appearance_weight: float = 0.4,
        max_epipolar_dist: float = 10.0,
        reid_threshold: float = 0.6,
        use_temporal: bool = True
    ):
        self.camera_manager = camera_manager
        self.triangulator = triangulator

        self.geometric = GeometricAssociator(camera_manager, triangulator, max_epipolar_dist)
        self.appearance = AppearanceAssociator(reid_extractor, reid_threshold)

        self.geometric_weight = geometric_weight
        self.appearance_weight = appearance_weight

        self.use_temporal = use_temporal
        self.prev_groups: List[MatchedGroup] = []  # Группы с предыдущего кадра

    def fuse_cost(self, geo_cost: float, app_cost: float) -> float:
        """Фьюзион стоимостей."""
        return self.geometric_weight * geo_cost + self.appearance_weight * app_cost

    def associate(
        self,
        frames: Dict[int, np.ndarray],
        detections_per_cam: Dict[int, List[Detection2D]],
        frame_idx: int
    ) -> List[MatchedGroup]:
        """
        Основной метод ассоциации.
        1. Извлекаем эмбеддинги (если есть ReID)
        2. Строим комбинированную матрицу стоимости
        3. Кластеризуем / Hungarian
        4. Триангуляция для верификации
        5. Временная согласованность с предыдущим кадром
        """
        cam_order = sorted(detections_per_cam.keys())

        # 1. Appearance embeddings
        detections_per_cam = self.appearance.extract_embeddings(frames, detections_per_cam)

        # 2. Собираем все детекции
        all_dets = []
        for cam_id in cam_order:
            for local_idx, det in enumerate(detections_per_cam.get(cam_id, [])):
                all_dets.append((cam_id, local_idx, det))

        n = len(all_dets)
        if n < 2:
            return []

        # 3. Cost matrices
        geo_cost, _ = self.geometric.build_cost_matrix(detections_per_cam, cam_order)
        app_cost = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                cam_i, _, det_i = all_dets[i]
                cam_j, _, det_j = all_dets[j]
                if cam_i != cam_j:
                    app_cost[i, j] = self.appearance.appearance_cost(det_i, det_j)
                    app_cost[j, i] = app_cost[i, j]

        # Нормализация геометрической стоимости (приводим к [0, 1])
        max_geo = self.geometric.max_epipolar_dist
        geo_norm = np.clip(geo_cost / max_geo, 0, 1)

        # Фьюжн
        fused_cost = self.geometric_weight * geo_norm + self.appearance_weight * app_cost
        np.fill_diagonal(fused_cost, 0)

        # 4. Кластеризация / Hungarian
        # Для простоты используем жадную кластеризацию
        groups = self._greedy_clustering(all_dets, fused_cost, frame_idx)

        # 5. Временная согласованность
        if self.use_temporal and self.prev_groups:
            groups = self._temporal_consistency(groups)

        self.prev_groups = groups
        return groups

    def _greedy_clustering(
        self,
        all_dets: List[Tuple[int, int, Detection2D]],
        cost_matrix: np.ndarray,
        frame_idx: int,
        threshold: float = 0.5
    ) -> List[MatchedGroup]:
        """Жадная кластеризация по фьюзионной стоимости."""
        n = len(all_dets)
        used = np.zeros(n, dtype=bool)
        groups = []

        for i in range(n):
            if used[i]:
                continue
            cam_i, local_i, det_i = all_dets[i]

            group_dets = {cam_i: det_i}
            used[i] = True

            for j in range(i+1, n):
                if used[j]:
                    continue
                cam_j, local_j, det_j = all_dets[j]
                if cam_j in group_dets:
                    continue

                if cost_matrix[i, j] < threshold:
                    group_dets[cam_j] = det_j
                    used[j] = True

            if len(group_dets) >= 2:
                # Триангуляция
                obs_list = []
                for cam_id, det in group_dets.items():
                    obs_list.append(Observation2D(
                        cam_id=cam_id,
                        u=det.center[0],
                        v=det.center[1],
                        confidence=det.confidence,
                        track_id=det.track_id,
                        frame_idx=frame_idx
                    ))

                tri_result = self.triangulator.triangulate_single(obs_list, frame_idx, use_ransac=False)
                if tri_result and tri_result.success:
                    groups.append(MatchedGroup(
                        detections=group_dets,
                        triangulation=tri_result,
                        confidence=np.mean([d.confidence for d in group_dets.values()])
                    ))

        return groups

    def _temporal_consistency(self, current_groups: List[MatchedGroup]) -> List[MatchedGroup]:
        """Согласование с предыдущим кадром по 3D позиции и ID."""
        if not self.prev_groups:
            return current_groups

        # Матрица стоимости между текущими и предыдущими группами
        n_curr = len(current_groups)
        n_prev = len(self.prev_groups)
        if n_curr == 0 or n_prev == 0:
            return current_groups

        cost = np.full((n_curr, n_prev), 1e6)
        for i, g_curr in enumerate(current_groups):
            if g_curr.triangulation is None:
                continue
            pos_curr = g_curr.triangulation.point_3d
            for j, g_prev in enumerate(self.prev_groups):
                if g_prev.triangulation is None:
                    continue
                pos_prev = g_prev.triangulation.point_3d
                dist = np.linalg.norm(pos_curr - pos_prev)
                cost[i, j] = dist

        # Hungarian для назначения ID
        row_ind, col_ind = linear_sum_assignment(cost)

        # Создаём новые группы с перенесёнными ID
        new_groups = []
        used_prev = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 500:  # порог 5 метров
                g_curr = current_groups[r]
                g_prev = self.prev_groups[c]
                g_curr.track_id_3d = g_prev.track_id_3d
                new_groups.append(g_curr)
                used_prev.add(c)

        # Остальные текущие группы без совпадения — новые ID
        for i, g in enumerate(current_groups):
            if i not in row_ind:
                new_groups.append(g)

        return new_groups

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику ассоциатора."""
        return {
            "geometric_weight": self.geometric_weight,
            "appearance_weight": self.appearance_weight,
            "use_temporal": self.use_temporal,
            "prev_groups_count": len(self.prev_groups)
        }


# =============================================================================
# Factory
# =============================================================================

def create_associator(
    camera_manager: CameraManager,
    triangulator,
    config_path: str = None
) -> HybridAssociator:
    """Фабрика для создания ассоциатора."""
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    assoc_config = config.get('association', {})
    reid_model = assoc_config.get('reid_model')

    reid_extractor = None
    if reid_model and Path(reid_model).exists():
        reid_extractor = ReIDExtractor(reid_model)

    return HybridAssociator(
        camera_manager=camera_manager,
        triangulator=triangulator,
        reid_extractor=reid_extractor,
        geometric_weight=assoc_config.get('geometric_weight', 0.6),
        appearance_weight=assoc_config.get('appearance_weight', 0.4),
        max_epipolar_dist=assoc_config.get('max_epipolar_distance_px', 10.0),
        reid_threshold=assoc_config.get('reid_threshold', 0.6),
        use_temporal=assoc_config.get('use_temporal_consistency', True)
    )