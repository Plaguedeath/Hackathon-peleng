"""
train_yolo.py
============================================================
Обучение модели детекции объекта на видео (YOLOv8).

Обучает модель на автоматически размеченном датасете (dataset/).
Поскольку в системе есть только CPU (torch+cpu), обучение будет
медленным — для начала достаточно небольшого числа эпох.

Используемая модель: YOLOv8n (nano) — самая быстрая и лёгкая.

Запуск:
    python train_yolo.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_YAML = ROOT / "dataset" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"

MODEL_NAME = "yolov8n.pt"   # nano — минимум ресурсов
EPOCHS = 20                 # для CPU — достаточно для проверки; можно увеличить
IMG_SIZE = 320
BATCH = 1                   # минимум из-за критически мало памяти (1 GB free)
WORKERS = 0


def main() -> None:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"не найден {DATASET_YAML}. Сначала запустите prepare_dataset.py"
        )

    print(f"датасет: {DATASET_YAML}")
    print(f"модель: {MODEL_NAME}, эпохи: {EPOCHS}, размер: {IMG_SIZE}, батч: {BATCH}")
    print("обучение на CPU — может занять много времени...")

    # загружаем предобученную модель
    model = YOLO(MODEL_NAME)

    # обучение
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device="cpu",
        project=str(RUNS_DIR),
        name="peleng_detect",
        exist_ok=True,
        verbose=True,
        amp=False,  # отключаем AMP для экономии памяти
    )

    # валидация на выборке val
    metrics = model.val(data=str(DATASET_YAML))
    print("\n=== Метрики на валидации ===")
    print(f"mAP50:  {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")

    best_model = RUNS_DIR / "peleng_detect" / "weights" / "best.pt"
    print(f"\nлучшая модель: {best_model}")
    print("\nОбучение завершено. Используйте best.pt для детекции на видео.")


if __name__ == "__main__":
    main()
