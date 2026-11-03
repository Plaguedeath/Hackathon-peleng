"""
prepare_dataset.py
============================================================
Подготовка датасета для обучения модели детекции объекта (YOLO).

Извлекает кадры из 5 видео и подбирает к ним автоматические метки
(labels_2d.csv, созданные auto_label.py). Сохраняет:

  dataset/
    images/train, images/val   — извлечённые кадры
    labels/train, labels/val   — YOLO-разметка (.txt)
    dataset.yaml               — конфиг датасета

Запуск:
    python prepare_dataset.py
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2

from camera_calib import VIDEO_WIDTH, VIDEO_HEIGHT

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "dataset"

# camera_id -> видеофайл
VIDEO_FILES = {
    1: "Way1.FinalImage.CineCameraActor1.mp4",
    2: "Way1.FinalImage.CineCameraActor2.mp4",
    3: "Way1.FinalImage.CineCameraActor3.mp4",
    4: "Way1.FinalImage.CineCameraActor4.mp4",
    5: "Way1.FinalImage.CineCameraActor5.mp4",
}

LABELS_CSV = DATA_DIR / "labels_2d.csv"
VAL_SPLIT = 0.2   # доля валидации
SEED = 42


def load_labeled_frames() -> list[dict]:
    """Читает labels_2d.csv и возвращает только в-кадровые метки."""
    items = []
    with open(LABELS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["in_frame"] == "True":
                items.append(
                    {
                        "frame": int(row["frame"]),
                        "camera_id": int(row["camera_id"]),
                        "u": float(row["u"]),
                        "v": float(row["v"]),
                        "focal_mm": float(row["focal_mm"]),
                    }
                )
    return items


def extract_frames(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Извлекает кадры из видео, сохраняет изображения и YOLO-метки.
    Возвращает (train_items, val_items).
    """
    random.seed(SEED)
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_SPLIT))
    val_items = items[:n_val]
    train_items = items[n_val:]

    # очистка папок датасета
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # кэш открытых видео, чтобы не переоткрывать на каждом кадре
    caps: dict[int, cv2.VideoCapture] = {}

    def get_cap(camera_id: int) -> cv2.VideoCapture:
        if camera_id not in caps:
            path = str(DATA_DIR / VIDEO_FILES[camera_id])
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"не удалось открыть видео {path}")
            caps[camera_id] = cap
        return caps[camera_id]

    def save_item(item: dict, split: str) -> None:
        cap = get_cap(item["camera_id"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
        ok, img = cap.read()
        if not ok:
            print(f"  [пропуск] не удалось прочитать кадр {item['frame']} камеры {item['camera_id']}")
            return

        # имя файла уникально: cam{camera_id}_frame{frame}
        stem = f"cam{item['camera_id']}_frame{item['frame']:04d}"
        img_path = DATASET_DIR / "images" / split / f"{stem}.jpg"
        label_path = DATASET_DIR / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(img_path), img)

        # YOLO-разметка: один класс (объект), бокс 50x50 вокруг проекции
        w, h = VIDEO_WIDTH, VIDEO_HEIGHT
        x_c = item["u"] / w
        y_c = item["v"] / h
        bw = 50.0 / w
        bh = 50.0 / h
        x_c = min(max(x_c, 0.0), 1.0)
        y_c = min(max(y_c, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        label_path.write_text(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    print(f"всего размеченных кадров: {len(items)} (train={len(train_items)}, val={n_val})")
    for split, subset in (("train", train_items), ("val", val_items)):
        print(f"извлечение {split}...")
        for item in subset:
            save_item(item, split)

    for cap in caps.values():
        cap.release()

    return train_items, val_items


def write_dataset_yaml(nc: int = 1, names: list[str] | None = None) -> Path:
    if names is None:
        names = ["object"]
    yaml_path = DATASET_DIR / "dataset.yaml"
    content = (
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"датасет конфиг: {yaml_path}")
    return yaml_path


def main() -> None:
    items = load_labeled_frames()
    print(f"загружено меток: {len(items)}")
    train_items, val_items = extract_frames(items)
    write_dataset_yaml()
    print("\nДатасет подготовлен. Следующий шаг — обучение (train_yolo.py).")


if __name__ == "__main__":
    main()
