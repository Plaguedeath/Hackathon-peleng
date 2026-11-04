"""
3D Multi-Object Tracker — трекинг объектов в 3D пространстве.
Использует Kalman Filter (CV/CA модели) с управлением треками (birth/death/occlusion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .triangulator import TriangulationResult


# =============================================================================
# Kalman Filter
# =============================================================================

class MotionModel(Enum):
    CV = "CV"      # Constant Velocity
    CA = "CA"      # Constant Acceleration


@dataclass
class KalmanFilter3D:
    """
    Kalman Filter для 3D трекинга.
    State: [x, y, z, vx, vy, vz] для CV
    State: [x, y, z, vx, vy, vz, ax, ay, az] для CA
    """
    dt: float = 1/30
    motion_model: MotionModel = MotionModel.CA
    process_noise_pos: float = 0.1
    process_noise_vel: float = 1.0
    process_noise_acc: float = 5.0

    # State и ковариация
    x: np.ndarray = field(default_factory=lambda: np.zeros(9))  # state
    P: np.ndarray = field(default_factory=lambda: np.eye(9) * 1000)  # covariance

    # Матрицы (инициализируются в _build_matrices)
    F: np.ndarray = field(default_factory=lambda: np.eye(9))
    Q: np.ndarray = field(default_factory=lambda: np.eye(9))
    H: np.ndarray = field(default_factory=lambda: np.zeros((3, 9)))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self._build_matrices()

    def _build_matrices(self):
        dt = self.dt
        if self.motion_model == MotionModel.CV:
            # State: [x, y, z, vx, vy, vz] (6)
            self.F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=float)
            self.Q = np.diag([
                self.process_noise_pos, self.process_noise_pos, self.process_noise_pos,
                self.process_noise_vel, self.process_noise_vel, self.process_noise_vel
            ])
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ], dtype=float)
            self.R = np.eye(3)
            self.x = np.zeros(6)
            self.P = np.eye(6) * 1000
        else:
            # CA: State: [x, y, z, vx, vy, vz, ax, ay, az] (9)
            dt2 = dt * dt / 2
            self.F = np.array([
                [1, 0, 0, dt, 0, 0, dt2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=float)
            self.Q = np.diag([
                self.process_noise_pos, self.process_noise_pos, self.process_noise_pos,
                self.process_noise_vel, self.process_noise_vel, self.process_noise_vel,
                self.process_noise_acc, self.process_noise_acc, self.process_noise_acc
            ])
            self.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0]
            ], dtype=float)
            self.R = np.eye(3)
            self.x = np.zeros(9)
            self.P = np.eye(9) * 1000

    def init_state(self, position: np.ndarray, velocity: np.ndarray = None, acceleration: np.ndarray = None):
        """Инициализация состояния."""
        if self.motion_model == MotionModel.CV:
            self.x[:3] = position
            self.x[3:6] = velocity if velocity is not None else np.zeros(3)
        else:
            self.x[:3] = position
            self.x[3:6] = velocity if velocity is not None else np.zeros(3)
            self.x[6:9] = acceleration if acceleration is not None else np.zeros(3)
        # Начальная ковариация позиции меньше
        self.P[:3, :3] = np.eye(3) * 10.0

    def predict(self):
        """Предсказание следующего состояния."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray = None):
        """Коррекция по измерению."""
        if measurement_cov is None:
            R = self.R
        else:
            R = measurement_cov

        # Innovation
        z = measurement.reshape(-1, 1)
        y = z - self.H @ self.x.reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + R

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + (K @ y).flatten()
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return y.flatten(), S

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        if self.motion_model == MotionModel.CV:
            return self.x[3:6].copy()
        return self.x[3:6].copy()

    def get_state_vector(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def mahalanobis_distance(self, measurement: np.ndarray, measurement_cov: np.ndarray = None) -> float:
        """Расстояние Махаланобиса до измерения."""
        if measurement_cov is None:
            R = self.R
        else:
            R = measurement_cov
        z = measurement.reshape(-1, 1)
        y = z - self.H @ self.x.reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + R
        mahal = y.T @ np.linalg.inv(S) @ y
        return float(mahal.item() if hasattr(mahal, 'item') else mahal[0, 0])


# =============================================================================
# Track State
# =============================================================================

class TrackState(Enum):
    TENTATIVE = "tentative"      # Кандидат, не подтверждён
    CONFIRMED = "confirmed"      # Подтверждённый активный трек
    LOST = "lost"                # Потерян (пропущено несколько кадров)
    DELETED = "deleted"          # Удалён


@dataclass
class Track3D:
    """3D трек объекта."""
    track_id: int
    kf: KalmanFilter3D
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1               # Количество успешных обновлений
    misses: int = 0             # Количество пропусков подряд
    age: int = 0                # Возраст в кадрах
    last_update_frame: int = 0
    # История для визуализации и анализа
    history_3d: List[np.ndarray] = field(default_factory=list)
    history_cov: List[np.ndarray] = field(default_factory=list)
    # Метаданные
    class_id: int = -1
    class_name: str = "unknown"
    # 2D track IDs на камерах (для связи назад)
    cam_track_ids: Dict[int, int] = field(default_factory=dict)

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update(self, measurement: np.ndarray, meas_cov: np.ndarray,
               frame_idx: int, cam_id: int = None, cam_track_id: int = None):
        self.kf.update(measurement, meas_cov)
        self.hits += 1
        self.misses = 0
        self.last_update_frame = frame_idx
        if cam_id is not None and cam_track_id is not None:
            self.cam_track_ids[cam_id] = cam_track_id

        # История
        self.history_3d.append(self.kf.get_position().copy())
        self.history_cov.append(self.kf.get_covariance().copy())

        # Перевод состояний
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def miss(self):
        self.misses += 1
        if self.state == TrackState.CONFIRMED and self.misses >= 5:
            self.state = TrackState.LOST

    def mark_deleted(self):
        self.state = TrackState.DELETED

    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def get_position(self) -> np.ndarray:
        return self.kf.get_position()

    def get_velocity(self) -> np.ndarray:
        return self.kf.get_velocity()


# =============================================================================
# 3D Multi-Object Tracker
# =============================================================================

class MultiObjectTracker3D:
    """
    Менеджер 3D треков.
    Ассоциирует триангулированные 3D точки с существующими треками.
    """

    def __init__(
        self,
        motion_model: MotionModel = MotionModel.CA,
        dt: float = 1/30,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
        process_noise_acc: float = 5.0,
        measurement_noise: float = 0.5,
        gate_threshold: float = 9.21,       # chi2(0.99, dof=3)
        max_misses: int = 30,
        min_hits: int = 3,
        max_age: int = 1000,
        iou_threshold_3d: float = 0.3
    ):
        self.motion_model = motion_model
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_acc = process_noise_acc
        self.measurement_noise = measurement_noise
        self.gate_threshold = gate_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold_3d = iou_threshold_3d

        self.tracks: Dict[int, Track3D] = {}
        self.next_id = 0
        self.frame_count = 0

        # Статистика
        self.stats = {
            'total_tracks_created': 0,
            'total_tracks_deleted': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0
        }

    def _create_kf(self) -> KalmanFilter3D:
        return KalmanFilter3D(
            dt=self.dt,
            motion_model=self.motion_model,
            process_noise_pos=self.process_noise_pos,
            process_noise_vel=self.process_noise_vel,
            process_noise_acc=self.process_noise_acc
        )

    def _create_track(self, measurement: np.ndarray, meas_cov: np.ndarray,
                      frame_idx: int, class_id: int = -1, class_name: str = "unknown") -> Track3D:
        kf = self._create_kf()
        kf.init_state(measurement)
        # Настройка R на основе measurement_cov
        kf.R = meas_cov if meas_cov is not None else np.eye(3) * self.measurement_noise

        track = Track3D(
            track_id=self.next_id,
            kf=kf,
            class_id=class_id,
            class_name=class_name
        )
        track.update(measurement, kf.R, frame_idx)
        self.next_id += 1
        self.stats['total_tracks_created'] += 1
        return track

    def _compute_cost_matrix(
        self,
        tracks: List[Track3D],
        measurements: List[TriangulationResult]
    ) -> np.ndarray:
        """Строит матрицу стоимости для Hungarian algorithm.
        Использует расстояние Махаланобиса + штраф за неопределённость.
        """
        n_tracks = len(tracks)
        n_meas = len(measurements)
        if n_tracks == 0 or n_meas == 0:
            return np.empty((n_tracks, n_meas))

        cost = np.full((n_tracks, n_meas), 1e6)

        for i, track in enumerate(tracks):
            for j, meas in enumerate(measurements):
                # Gating: проверяем расстояние Махаланобиса
                dist = track.kf.mahalanobis_distance(meas.point_3d, meas.covariance)
                if dist > self.gate_threshold:
                    continue

                # Стоимость = расстояние Махаланобиса
                cost[i, j] = dist

        return cost

    def step(
        self,
        triangulations: List[TriangulationResult],
        frame_idx: int
    ) -> List[Track3D]:
        """
        Основной шаг трекера:
        1. Predict все треки
        2. Ассоциируем измерения с треками (Hungarian)
        3. Обновляем совпавшие, создаём новые, помечаем пропуски
        4. Удаляем старые потерянные треки
        Возвращает список активных подтверждённых треков.
        """
        self.frame_count = frame_idx

        # 1. Predict
        for track in self.tracks.values():
            if track.is_active():
                track.predict()

        # 2. Фильтруем активные треки для ассоциации
        active_tracks = [t for t in self.tracks.values() if t.is_active()]
        measurements = [t for t in triangulations if t.success]

        # 3. Association
        if active_tracks and measurements:
            cost = self._compute_cost_matrix(active_tracks, measurements)
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_meas = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1e6:  # Валидное совпадение
                    track = active_tracks[r]
                    meas = measurements[c]
                    track.update(meas.point_3d, meas.covariance, frame_idx)
                    matched_tracks.add(track.track_id)
                    matched_meas.add(c)
                else:
                    # Слишком большое расстояние — пропускаем
                    pass

            # 4. Обработка несопоставленных измерений -> новые треки
            for j, meas in enumerate(measurements):
                if j not in matched_meas:
                    # Определяем класс (например, по первому наблюдению)
                    class_id = -1
                    class_name = "unknown"
                    if meas.observations:
                        class_id = meas.observations[0].cam_id  # placeholder
                    new_track = self._create_track(meas.point_3d, meas.covariance, frame_idx,
                                                   class_id, class_name)
                    self.tracks[new_track.track_id] = new_track

            # 5. Обработка несопоставленных треков -> miss
            for track in active_tracks:
                if track.track_id not in matched_tracks:
                    track.miss()

        elif measurements:
            # Нет активных треков, создаём новые из всех измерений
            for meas in measurements:
                class_id = -1
                class_name = "unknown"
                if meas.observations:
                    class_id = meas.observations[0].cam_id
                new_track = self._create_track(meas.point_3d, meas.covariance, frame_idx,
                                               class_id, class_name)
                self.tracks[new_track.track_id] = new_track

        else:
            # Нет измерений, все активные треки получают miss
            for track in active_tracks:
                track.miss()

        # 6. Удаление старых потерянных треков
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.state == TrackState.DELETED:
                to_delete.append(track_id)
            elif track.state == TrackState.LOST and track.misses >= self.max_misses:
                track.mark_deleted()
                to_delete.append(track_id)
            elif track.age > self.max_age:
                track.mark_deleted()
                to_delete.append(track_id)

        for tid in to_delete:
            del self.tracks[tid]
            self.stats['total_tracks_deleted'] += 1

        # 7. Обновление статистики
        self.stats['active_tracks'] = sum(1 for t in self.tracks.values() if t.is_active())
        self.stats['confirmed_tracks'] = sum(1 for t in self.tracks.values() if t.is_confirmed())

        # Возвращаем подтверждённые треки
        return [t for t in self.tracks.values() if t.is_confirmed()]

    def get_all_tracks(self) -> Dict[int, Track3D]:
        return self.tracks.copy()

    def get_confirmed_tracks(self) -> List[Track3D]:
        return [t for t in self.tracks.values() if t.is_confirmed()]

    def get_track(self, track_id: int) -> Optional[Track3D]:
        return self.tracks.get(track_id)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        for k in self.stats:
            self.stats[k] = 0


# =============================================================================
# Factory
# =============================================================================

def create_tracker3d(config_path: str = None) -> MultiObjectTracker3D:
    """Фабрика для создания 3D трекера."""
    import yaml
    from pathlib import Path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    t3d = config.get('tracker3d', {})
    pn = t3d.get('process_noise', {})
    if isinstance(pn, dict):
        process_noise_pos = pn.get('position', 0.1)
        process_noise_vel = pn.get('velocity', 1.0)
        process_noise_acc = pn.get('acceleration', 5.0)
    else:
        process_noise_pos = float(pn)
        process_noise_vel = float(pn)
        process_noise_acc = float(pn)
    return MultiObjectTracker3D(
        motion_model=MotionModel(t3d.get('motion_model', 'CA')),
        dt=t3d.get('dt', 1/30),
        process_noise_pos=process_noise_pos,
        process_noise_vel=process_noise_vel,
        process_noise_acc=process_noise_acc,
        measurement_noise=t3d.get('measurement_noise', 0.5),
        gate_threshold=t3d.get('gate_threshold', 9.21),
        max_misses=t3d.get('max_misses', 30),
        min_hits=t3d.get('min_hits', 3),
        max_age=t3d.get('max_age', 1000),
        iou_threshold_3d=t3d.get('iou_threshold_3d', 0.3)
    )


if __name__ == "__main__":
    # Тест
    tracker = create_tracker3d()

    # Имитация измерений движущейся точки
    for frame in range(100):
        # Линейное движение
        x = np.array([frame * 10.0, 100.0, 50.0])
        cov = np.eye(3) * 0.5
        from triangulator import TriangulationResult, Observation2D
        meas = TriangulationResult(
            point_3d=x,
            covariance=cov,
            observations=[],
            reprojection_errors={},
            success=True
        )
        tracks = tracker.step([meas], frame)

        if frame % 20 == 0:
            print(f"Frame {frame}: {len(tracks)} tracks")
            for tr in tracks:
                print(f"  ID {tr.track_id}: pos={tr.get_position()}, vel={tr.get_velocity()}")

    print(f"Stats: {tracker.get_stats()}")