# Peleng Multi-Camera 3D Tracking Pipeline

**Universal multi-object 3D tracker** built on YOLOv8 + BoTSORT + 3D Kalman filtering with epipolar geometry and ReID association.

## 🎯 Features

- **Universal Detection**: YOLOv8 (n/s/m/l/x) with configurable classes
- **Per-Camera 2D Tracking**: BoTSORT / ByteTrack via Ultralytics
- **Cross-Camera Association**: Epipolar geometry + OSNet ReID + temporal consistency
- **Robust Triangulation**: DLT / Optimal (Hartley-Sturm) / Midpoint + RANSAC
- **3D Kalman Filtering**: CV (Constant Velocity) & CA (Constant Acceleration) models
- **Track Management**: Tentative → Confirmed → Lost → Deleted state machine
- **Uncertainty Propagation**: 2D covariance → 3D covariance via projection Jacobian
- **Real-time Pipeline**: Async processing with ThreadPoolExecutor
- **Rich Visualization**: 2D overlays, 3D trajectory animation, metrics plots
- **Config-Driven**: Zero-code camera/setup changes via YAML/JSON

## 📁 Project Structure

```
peleng/
├── config/
│   ├── cameras.json      # Camera calibration (intrinsics/extrinsics from UE)
│   ├── tracker.yaml      # Main pipeline configuration
│   └── botsort.yaml      # BoTSORT tracker config
├── src/
│   ├── __init__.py       # Package exports
│   ├── camera_manager.py # Camera I/O, projection, UE coordinate handling
│   ├── detector.py       # YOLO + BoTSORT per-camera detection
│   ├── triangulator.py   # Multi-view triangulation with RANSAC
│   ├── associator.py     # Geometric + Appearance + Hybrid association
│   ├── tracker3d.py      # 3D Kalman filter + track management
│   ├── visualizer.py     # 2D/3D visualization, plots, video writer
│   └── pipeline.py       # Main orchestration (sync/async)
├── models/               # YOLO weights (yolov8n.pt, etc.)
├── data/                 # Video files
├── run.py                # CLI entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
cd peleng
pip install -r requirements.txt
```

### Download YOLO Model

```bash
# Place your trained model in models/
# Default: models/yolov8n.pt (downloaded automatically by ultralytics on first run)
```

### Configure Cameras

Edit `config/cameras.json` with your camera calibration (Unreal Engine format):

```json
{
  "cameras": [
    {
      "id": 1,
      "video_path": "data/cam1.mp4",
      "intrinsics": {
        "focal_mm": 24.0,
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "width": 1920,
        "height": 1080
      },
      "extrinsics": {
        "location_cm": [0, 0, 300],
        "rotation_deg": [0, -90, 0]
      }
    }
  ]
}
```

### Run Pipeline

```bash
# Basic run (sync mode)
python run.py

# Process only 100 frames
python run.py --max-frames 100

# Async mode with 4 workers (faster on multi-core)
python run.py --async --workers 4

# Disable visualization for max speed
python run.py --no-vis

# Custom config
python run.py -c config/my_config.yaml

# List available cameras
python run.py --list-cameras
```

## ⚙️ Configuration

### `config/tracker.yaml` — Main Pipeline Config

```yaml
detection:
  model_path: "models/yolov8n.pt"
  conf_threshold: 0.25
  iou_threshold: 0.45
  imgsz: 640
  device: "cpu"           # "cuda:0" for GPU
  tracker: "botsort"      # "botsort" | "bytetrack" | null
  tracker_config: "config/botsort.yaml"
  classes: null           # null = all classes, or [0, 2, 7] for specific COCO classes

triangulation:
  method: "optimal"       # "dlt" | "optimal" | "midpoint"
  ransac: true
  ransac_threshold_px: 2.0
  min_views: 2
  min_angle_deg: 1.0

tracker3d:
  motion_model: "CV"      # "CV" | "CA"
  dt: 0.033               # frame time (1/30)
  process_noise: 100.0
  measurement_noise: 50.0
  confirmation_frames: 3
  deletion_frames: 10
  max_lost_frames: 30
  mahalanobis_gate: 9.488 # chi2_0.99(4)

association:
  geometric_weight: 0.6
  appearance_weight: 0.4
  epipolar_threshold_px: 5.0
  appearance_threshold: 0.35
  use_temporal: true
  max_time_gap: 5

visualization:
  enabled: true
  save_video: true
  draw_2d_boxes: true
  draw_3d_reprojection: true
  draw_trajectory_3d: true
  draw_plots: true
  output_dir: "output"
  output_fps: 30
  max_trail_length: 100

pipeline:
  mode: "sync"            # "sync" | "async"
  async_workers: 2
  max_frames: -1          # -1 = all frames
  skip_frames: 0          # process every Nth frame
```

## 🔧 Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Camera 1   │     │  Camera 2   │ ... │  Camera N   │
│  (video)    │     │  (video)    │     │  (video)    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│           MultiCameraDetector (YOLO + BoTSORT)      │
│  Per-camera 2D detection + tracking → track_id_2d  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   Triangulator                       │
│  Epipolar geometry + RANSAC → 3D points + covariance│
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   HybridAssociator                   │
│  Geometric (epipolar) + Appearance (OSNet ReID)     │
│  + Temporal consistency → cross-camera matches      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              MultiObjectTracker3D                    │
│  3D Kalman Filter (CV/CA) + Hungarian assignment    │
│  Track state machine: Tentative→Confirmed→Lost→Del  │
└────────────────────────┬────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      ┌─────────┐  ┌──────────┐  ┌──────────┐
      │ 2D Overlay│  │ 3D Traj. │  │  Plots   │
      │ (per cam) │  │ Animation│  │ (X/Y/Z-t)│
      └─────────┘  └──────────┘  └──────────┘
```

## 📊 Output

Results saved to `output/` (configurable):

```
output/
├── cam1_overlay.mp4      # 2D boxes + 3D reprojection
├── cam2_overlay.mp4
├── ...
├── trajectory_3d.gif     # 3D animated trajectories
├── trajectory_3d_static.png
├── trajectories_vs_time.png
├── velocity_vs_time.png
├── reprojection_errors.png
├── metrics_summary.png
└── metrics.json          # All numeric metrics
```

## 🎯 Key Classes

| Module | Class | Purpose |
|--------|-------|---------|
| `camera_manager` | `CameraManager` | Video I/O, UE projection, ray casting |
| `detector` | `MultiCameraDetector` | YOLO + BoTSORT per camera |
| `triangulator` | `Triangulator` | DLT/Optimal/Midpoint + RANSAC |
| `associator` | `HybridAssociator` | Epipolar + ReID + Temporal fusion |
| `tracker3d` | `MultiObjectTracker3D` | 3D Kalman + Hungarian + State machine |
| `visualizer` | `MultiCameraVisualizer` | 2D/3D plots, video, animations |
| `pipeline` | `MultiCameraTrackingPipeline` | Sync/Async orchestration |

## 🧪 Testing

```bash
# Test camera manager
python -c "from src.camera_manager import create_camera_manager; cm=create_camera_manager(); cm.open_all(); print('Cams:', cm.get_camera_ids()); cm.close_all()"

# Test detector
python -m src.detector

# Test triangulator
python -m src.triangulator

# Test full pipeline (10 frames)
python run.py --max-frames 10 --no-vis
```

## 📝 Retraining YOLO

For better detection (current model undertrained):

```bash
# 1. Prepare data in YOLO format (images + labels)
# 2. Create data.yaml with train/val paths and class names
# 3. Train:
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16 device=0 amp=True

# 4. Copy best.pt to models/
cp runs/detect/train/weights/best.pt models/yolov8s_custom.pt

# 5. Update config/tracker.yaml model_path
```

Recommended: `yolov8s` or `yolov8m`, 640px, 100 epochs, AMP, COCO pretrained.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Cam3 0 detections | Check video path in `cameras.json`, verify labels overlap with FOV |
| 3D error > 100m | Verify calibration (UE coords: X-forward, Z-up, roll inverted), check baseline |
| Slow FPS | Use GPU (`device: cuda:0`), enable `--async`, reduce `imgsz`, disable vis |
| BoTSORT not tracking | Ensure `tracker: botsort` and `tracker_config` points to valid YAML |

## 📄 License

Internal research project — Peleng.

---

**Built for multi-camera 3D tracking with Unreal Engine calibration data.**