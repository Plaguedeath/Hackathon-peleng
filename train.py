from ultralytics import YOLO

# Загружаем предобученную модель
model = YOLO("yolov8s.pt")

# Запускаем обучение с вашими параметрами
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,      # Использовать GPU (CUDA device 0)
    amp=True       # Включить Automatic Mixed Precision
)