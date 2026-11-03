"""
train_custom_yolo.py
============================================================
Кастомное обучение YOLO на конкретном грузовике.

Использует GT траекторию (object_transform_SM_AA_track.csv) для автоматической
разметки кадров на всех 5 камерах. Проецирует 3D точку грузовика (с Z-offset)
на 2D пиксели каждой камеры → получает идеальные YOLO боксы без ручной разметки.

Выход: кастомная модель в models/yolov8s_truck.pt
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO

from src.camera_manager import create_camera_manager

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"
DATASET_DIR = RUNS_DIR / "dataset_custom"

# Параметры
IMG_SIZE = 640
BATCH = 4  # CPU needs smaller batch
EPOCHS = 100
PATIENCE = 20
DEVICE = "cpu"  # CPU-only environment
HALF = False

# GT Z-offset (поднимаем точку от поворота до визуального центра кабины)
GT_Z_OFFSET_CM = 350.0

# Размер бокса вокруг проекции (покрывает весь грузовик)
BOX_HALF_W = 120  # половина ширины бокса в пикселях
BOX_HALF_H = 80   # половина высоты бокса в пикселях

VAL_SPLIT = 0.15
SEED = 42


def load_gt_trajectory() -> dict[int, np.ndarray]:
    """Загружает GT траекторию с Z-offset."""
    gt_path = DATA_DIR / "object_transform_SM_AA_track.csv"
    gt = {}
    with open(gt_path, "r", encoding="cp1251", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            try:
                frame = int(row[0])
                x = float(row[1].replace(",", "."))
                y = float(row[2].replace(",", "."))
                z = float(row[3].replace(",", ".")) + GT_Z_OFFSET_CM  # <-- Z-OFFSET
                gt[frame] = np.array([x, y, z], dtype=float)
            except (ValueError, IndexError):
                continue
    return gt


def project_gt_to_all_cameras(
    gt_trajectory: dict[int, np.ndarray],
    camera_manager,
    frames_to_process: list[int]
) -> dict[int, dict[int, tuple[float, float]]]:
    """
    Проецирует GT точку на все камеры для заданных кадров.
    Возвращает: {frame: {cam_id: (u, v)}}
    """
    projections = {}

    for frame in frames_to_process:
        if frame not in gt_trajectory:
            continue

        point_3d = gt_trajectory[frame]
        frame_projs = {}

        for cam_id in camera_manager.get_camera_ids():
            uv = camera_manager.project_world_to_image(cam_id, point_3d, frame)
            if uv is not None:
                u, v = uv
                # Проверяем, что проекция в пределах кадра с запасом
                cam = camera_manager.get_camera(cam_id)
                if 0 <= u < cam.intrinsics.width and 0 <= v < cam.intrinsics.height:
                    frame_projs[cam_id] = (u, v)

        if frame_projs:
            projections[frame] = frame_projs

    return projections


def create_yolo_dataset(
    projections: dict[int, dict[int, tuple[float, float]]],
    camera_manager,
    dataset_dir: Path
) -> tuple[list[dict], list[dict]]:
    """
    Создаёт YOLO датасет: извлекает кадры из видео, создаёт метки.
    Возвращает (train_items, val_items).
    """
    random.seed(SEED)
    np.random.seed(SEED)

    # Очистка и создание папок
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Группируем проекции по камерам для эффективного извлечения кадров
    frames_by_cam: dict[int, set[int]] = {}
    for frame, cam_projs in projections.items():
        for cam_id in cam_projs:
            frames_by_cam.setdefault(cam_id, set()).add(frame)

    label_entries = []

    print(f"Извлечение кадров и создание меток...")
    for cam_id, frames in frames_by_cam.items():
        frames_sorted = sorted(frames)
        cam = camera_manager.get_camera(cam_id)

        # Открываем видео
        cap = cv2.VideoCapture(cam.source)
        if not cap.isOpened():
            print(f"  Не удалось открыть {cam.source}")
            continue

        # Читаем только нужные кадры
        max_frame = max(frames_sorted)
        frame_idx = 0
        for target_frame in frames_sorted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"  Cam{cam_id} frame {target_frame}: не прочитан")
                continue

            # Сохраняем изображение
            img_name = f"cam{cam_id}_frame{target_frame:06d}.jpg"
            img_path = dataset_dir / "images" / "train" / img_name  # временно в train
            cv2.imwrite(str(img_path), frame)

            # Создаём YOLO метки для всех камер, видящих этот кадр
            for frame_check, cam_projs in projections.items():
                if frame_check != target_frame:
                    continue
                for proj_cam_id, (u, v) in cam_projs.items():
                    # Нормализованные координаты YOLO
                    w, h = cam.intrinsics.width, cam.intrinsics.height
                    xc = u / w
                    yc = v / h
                    bw = (2 * BOX_HALF_W) / w
                    bh = (2 * BOX_HALF_H) / h
                    xc = np.clip(xc, 0.0, 1.0)
                    yc = np.clip(yc, 0.0, 1.0)
                    bw = np.clip(bw, 0.0, 1.0)
                    bh = np.clip(bh, 0.0, 1.0)

                    label_path = dataset_dir / "labels" / "train" / f"cam{proj_cam_id}_frame{target_frame:06d}.txt"
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(label_path, "w") as f:
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                    label_entries.append({
                        "cam_id": proj_cam_id,
                        "frame": target_frame,
                        "img_name": f"cam{proj_cam_id}_frame{target_frame:06d}.jpg",
                        "label": f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                    })

        cap.release()
        print(f"  Cam{cam_id}: извлечено {len(frames_sorted)} кадров")

    # Split train/val
    random.shuffle(label_entries)
    n_val = max(1, int(len(label_entries) * VAL_SPLIT))
    val_entries = label_entries[:n_val]
    train_entries = label_entries[n_val:]

    # Перемещаем val файлы
    for entry in val_entries:
        src_img = dataset_dir / "images" / "train" / entry["img_name"]
        dst_img = dataset_dir / "images" / "val" / entry["img_name"]
        src_lbl = dataset_dir / "labels" / "train" / entry["img_name"].replace(".jpg", ".txt")
        dst_lbl = dataset_dir / "labels" / "val" / entry["img_name"].replace(".jpg", ".txt")

        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    print(f"\nДатасет создан: train={len(train_entries)}, val={len(val_entries)}")
    return train_entries, val_entries


def create_data_yaml(dataset_dir: Path, nc: int = 1, names: list = None):
    """Создаёт data.yaml для YOLO."""
    if names is None:
        names = ["truck"]

    yaml_content = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"data.yaml создан: {yaml_path}")
    return yaml_path


def train_yolo(
    data_yaml: Path,
    model_path: str = "models/yolov8s.pt",
    epochs: int = EPOCHS,
    imgsz: int = IMG_SIZE,
    batch: int = BATCH,
    device: str = DEVICE,
    half: bool = HALF,
    patience: int = PATIENCE
):
    """Обучает YOLO модель."""
    print(f"\n=== ОБУЧЕНИЕ YOLO ===")
    print(f"Модель: {model_path}")
    print(f"Эпохи: {epochs}, imgsz: {imgsz}, batch: {batch}")
    print(f"Device: {device}, half: {half}")

    model = YOLO(model_path)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        patience=patience,
        project=str(RUNS_DIR / "train_custom"),
        name="truck_detector",
        exist_ok=True,
        verbose=True,
        plots=True,
        save_period=10
    )

    best_model = RUNS_DIR / "train_custom" / "truck_detector" / "weights" / "best.pt"
    final_model = MODELS_DIR / "yolov8s_truck.pt"

    if best_model.exists():
        shutil.copy2(best_model, final_model)
        print(f"\n✅ Лучшая модель сохранена: {final_model}")
    else:
        print(f"\n⚠️ best.pt не найден, ищем last.pt...")
        last_model = RUNS_DIR / "train_custom" / "truck_detector" / "weights" / "last.pt"
        if last_model.exists():
            shutil.copy2(last_model, final_model)
            print(f"Сохранена last.pt: {final_model}")

    return final_model


def validate_on_videos(model_path: Path, camera_manager, gt_trajectory: dict):
    """Валидация модели на видео с подсчётом репроекционных ошибок."""
    print(f"\n=== ВАЛИДАЦИЯ НА ВИДЕО ===")
    model = YOLO(str(model_path))

    total_reproj_errors = []

    for cam_id in camera_manager.get_camera_ids():
        cam = camera_manager.get_camera(cam_id)
        cap = cv2.VideoCapture(cam.source)
        if not cap.isOpened():
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in gt_trajectory:
                # Проецируем GT
                gt_uv = camera_manager.project_world_to_image(cam_id, gt_trajectory[frame_idx], frame_idx)

                # Детекция
                results = model(frame, conf=0.25, verbose=False)
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cx, cy = box.xywh[0][:2].cpu().numpy()
                        if gt_uv:
                            err = np.sqrt((cx - gt_uv[0])**2 + (cy - gt_uv[1])**2)
                            total_reproj_errors.append(err)

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  Cam{cam_id}: {frame_idx} кадров...")

        cap.release()

    if total_reproj_errors:
        print(f"\nРепроекционная ошибка (валидация):")
        print(f"  Mean: {np.mean(total_reproj_errors):.2f} px")
        print(f"  RMSE: {np.sqrt(np.mean(np.array(total_reproj_errors)**2)):.2f} px")
        print(f"  Max: {np.max(total_reproj_errors):.2f} px")
        print(f"  N: {len(total_reproj_errors)}")


def main():
    print("=" * 60)
    print("КАСТОМНОЕ ОБУЧЕНИЕ YOLO НА ГРУЗОВИКЕ")
    print("=" * 60)

    # 1. Загружаем GT траекторию с Z-offset
    print(f"\n1. Загрузка GT траектории (Z-offset: {GT_Z_OFFSET_CM} см)...")
    gt_trajectory = load_gt_trajectory()
    print(f"   Загружено кадров: {len(gt_trajectory)}")

    # 2. Инициализируем CameraManager
    print("\n2. Инициализация камер...")
    camera_manager = create_camera_manager()
    camera_manager.open_all()

    # 3. Определяем кадры для обработки (где есть GT)
    frames_to_process = sorted(gt_trajectory.keys())
    print(f"   Кадров для обработки: {len(frames_to_process)}")

    # 4. Проецируем GT на все камеры
    print("\n3. Проекция GT на камеры...")
    projections = project_gt_to_all_cameras(gt_trajectory, camera_manager, frames_to_process)
    total_projs = sum(len(v) for v in projections.values())
    print(f"   Валидных проекций: {total_projs}")

    # Статистика по камерам
    cam_counts = {}
    for frame, projs in projections.items():
        for cam_id in projs:
            cam_counts[cam_id] = cam_counts.get(cam_id, 0) + 1
    for cam_id, count in sorted(cam_counts.items()):
        print(f"   Cam{cam_id}: {count} проекций")

    # 5. Создаём датасет
    print("\n4. Создание YOLO датасета...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_entries, val_entries = create_yolo_dataset(projections, camera_manager, DATASET_DIR)

    # 6. Создаём data.yaml
    print("\n5. Создание data.yaml...")
    data_yaml = create_data_yaml(DATASET_DIR)

    # 7. Обучаем модель
    print("\n6. Обучение модели...")
    best_model = train_yolo(data_yaml)

    # 8. Валидация
    print("\n7. Валидация на видео...")
    validate_on_videos(best_model, camera_manager, gt_trajectory)

    camera_manager.close_all()

    print(f"\n{'='*60}")
    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Модель: {best_model}")
    print(f"Датасет: {DATASET_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()