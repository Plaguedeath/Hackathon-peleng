# Hackathon-peleng — 3D Trajectory Reconstruction from Multi-Camera Video

Автоматический пайплайн для восстановления 3D-траектории объекта (фуры/грузовика) из 5 видеокамер с последующей валидацией против Ground Truth.

## 🎯 Суть задачи

Дано:
- 5 видео с камер (`Way1.FinalImage.CineCameraActor1..5.mp4`)
- CSV с параметрами камер (экстринсики + фокус) — `object_transform_CineCameraActor*.csv`
- Ground Truth 3D-траектория объекта — `object_transform_SM_AA_track.csv`

Цель: обучить детектор (YOLO), найти объект на каждом кадре, триангулировать 3D-точки, сгладить траекторию и сравнить с GT.

**Ключевое преимущество**: разметка вручную **не нужна** — 3D GT проецируется в 2D автоматически.

---

## 📁 Структура проекта

```
peleng/
├── data/                           # Исходные данные
│   ├── Way1.FinalImage.CineCameraActor1..5.mp4   # 5 видео
│   ├── object_transform_CineCameraActor1..5.csv  # Параметры камер
│   ├── object_transform_SM_AA_track.csv          # GT 3D траектория
│   ├── labels_2d.csv              # Автогенерация 2D меток (Этап 1)
│   └── yolo_labels/               # YOLO .txt файлы (Этап 1)
│
├── dataset/                        # Датасет для YOLO (Этап 2)
│   ├── images/train/, images/val/  # Извлечённые кадры
│   ├── labels/train/, labels/val/  # YOLO метки
│   └── dataset.yaml
│
├── runs/                           # Артефакты пайплайна
│   ├── peleng_detect/weights/best.pt    # Обученная модель
│   ├── detections/                  # Этап 3: all_detections.csv
│   ├── trajectory/                  # Этап 4-6: trajectory_*.csv
│   ├── validation/                  # Этап 6: отчёты, метрики
│   └── visualization/               # Этап 7: видео, GIF, графики
│
├── pipeline/                       # Альтернативная реализация (pure Python, без numpy)
│
├── camera_calib.py          # Ядро: калибровка, проекция 3D↔2D, триангуляция (numpy)
├── auto_label.py            # Этап 1: авторазметка из GT
├── prepare_dataset.py       # Этап 2: датасет для YOLO
├── train_yolo.py            # Этап 2: обучение YOLOv8n
├── run_inference.py         # Этап 3: инференс на 5 видео
├── triangulate.py           # Этап 4: триангуляция 3D
├── postprocess.py           # Этап 5: сглаживание (Сав-Голей, Калман)
├── validate.py              # Этап 6: валидация против GT
├── visualize.py             # Этап 7: визуализация
└── run_pipeline.py          # Мастер-скрипт всего пайплайна
```

---

## ⚙️ Установка

```bash
# Python 3.12+ (есть venv в .venv312/, .venv314/)
cd peleng

# Зависимости
pip install opencv-python numpy scipy matplotlib ultralytics torch torchvision pyyaml

# Или через requirements.txt (создайте при необходимости)
```

---

## 🚀 Быстрый старт (весь пайплайн)

```bash
# Запуск этапов 4-8 (инференс → триангуляция → постобработка → валидация → визуализация)
python run_pipeline.py

# Или по одному:
python run_inference.py          # Этап 3: детекции на видео
python triangulate.py            # Этап 4: 3D триангуляция
python postprocess.py            # Этап 5: сглаживание
python validate.py               # Этап 6: метрики против GT
python visualize.py              # Этап 7: видео, 3D GIF, графики
```

---

## 📋 Подробные этапы

### Этап 0: Параметры видео (опционально)
```bash
python pipeline/extract_video_meta.py
# → runs/metadata/video_meta.json
```

### Этап 1: Авторазметка из GT
```bash
python auto_label.py
# → data/labels_2d.csv, data/yolo_labels/*.txt
# Использует camera_calib.py для проекции 3D→2D
```

### Этап 2: Датасет + Обучение
```bash
python prepare_dataset.py
# → dataset/ (images/, labels/, dataset.yaml)

python train_yolo.py
# → runs/peleng_detect/weights/best.pt
# Параметры: epochs=20, imgsz=320, batch=1, CPU
```

### Этап 3: Инференс на видео
```bash
python run_inference.py [--conf 0.25] [--draw]
# → runs/detections/cam{N}_detections.csv + all_detections.csv
# --draw: сохраняет видео с нарисованными боксами
```

### Этап 4: Триангуляция
```bash
python triangulate.py
# → runs/trajectory/trajectory_raw.csv + triangulation_log.csv
# Использует камеры из camera_calib.py + детекции
```

### Этап 5: Постобработка
```bash
python postprocess.py
# → runs/trajectory/trajectory_smooth.csv + postprocess_log.csv
# Удаление выбросов → интерполяция → Savitzky-Golay → Kalman (forward-backward)
```

### Этап 6: Валидация
```bash
python validate.py [--traj runs/trajectory/trajectory_smooth.csv]
# → runs/validation/validation_report.csv (по-кадровая 3D ошибка)
# → runs/validation/reprojection_report.csv (репроекционная ошибка в px)
# → runs/validation/validation_summary.json (MAE, RMSE, Max, по осям)
```

### Этап 7: Визуализация
```bash
python visualize.py [--max-frames 500] [--no-videos] [--no-3d] [--no-plots]
# → runs/visualization/
#   cam{N}_overlay.mp4      — видео с YOLO боксами + репроекциями 3D
#   trajectory_3d.gif       — 3D анимация траектории + камеры
#   error_plots.png         — графики ошибок
#   summary_metrics.png     — сводный барплот метрик
```

---

## 📊 Ключевые метрики (Этап 6)

| Метрика | Описание |
|---------|----------|
| **MAE (3D)** | Средняя абсолютная ошибка траектории (см/м) |
| **RMSE (3D)** | RMS ошибка (чувствительна к выбросам) |
| **Max Error** | Максимальное отклонение от GT |
| **|dX|, |dY|, |dZ|** | Ошибки по осям |
| **Reproj MAE/RMSE** | Ошибка репроекции 3D→2D (пиксели) — качество калибровки |

---

## 🎥 Камеры (экстринсики из CSV)

| Камера | Loc (X,Y,Z) см | Rot (Roll,Pitch,Yaw)° | Focal | Видит объект |
|--------|----------------|----------------------|-------|--------------|
| Cam1 | (-24702, 6624, 1200) | (0, 0, -70) | 35 мм | ✅ |
| Cam2 | (-45784, -6500, 7912) | (0, -10, 0) | 60 мм | ❌ (узкий FOV) |
| Cam3 | (12600, -22700, 1400) | (-0.68, -3.11, 115.11) | 35 мм | ✅ |
| Cam4 | (33282, 13997, 7147) | (0, -10, -160) | 50 мм | ❌ (далеко) |
| Cam5 | (9799, -3900, 1488) | (-4.41, -6.97, yaw меняется) | 20 мм | ✅ (широкий FOV) |

> **Важно**: Cam5 вращается (yaw меняется покадрово) — экстринсики берутся по каждому кадру.

---

## 🔧 Полезные команды

```bash
# Только валидация (если траектория уже есть)
python validate.py --traj runs/trajectory/trajectory_raw.csv

# Только визуализация без видео (быстро)
python visualize.py --no-videos

# Полный пайплайн с этапа 5 (пропуск инференса/триангуляции)
python run_pipeline.py --from 5

# Только этап 6
python run_pipeline.py --only 6

# С рисованными боксами на этапе инференса
python run_inference.py --draw
```

---

## ⚠️ Риски и нюансы

1. **Конвенция координат UE** — левосторонняя, обзор камеры вдоль +X. Сверять с `SM_AA_track.csv`.
2. **Cam5 вращается** — используется покадровый yaw, остальные камеры статичны.
3. **Cam2, Cam4 не видят объект** — триангуляция только там, где ≥2 камеры видят одновременно.
4. **Окклузии** — объект может быть скрыт; пропуски интерполируются на Этапе 5.
5. **CSV кодировка** — cp1251, разделитель `;`, десятичные запятые (23,76).

---

## 📈 Ожидаемые результаты

После полного запуска в `runs/validation/validation_summary.json`:
```json
{
  "error_3d_cm": {
    "mean": 150.0,      // ~1.5 м средняя ошибка
    "rmse": 200.0,
    "max": 500.0,
    "mean_abs_dx": 100.0,
    "mean_abs_dy": 80.0,
    "mean_abs_dz": 50.0
  },
  "reprojection_error_px": {
    "mean": 8.5,        // < 10 px — хорошая калибровка
    "rmse": 12.0,
    "max": 35.0
  }
}
```

Хорошая калибровка даёт репроекционную ошибку **< 10-15 px**. 3D ошибка зависит от базы камер и качества детекций.

---

## 📝 Лицензия

Hackathon project — internal use.