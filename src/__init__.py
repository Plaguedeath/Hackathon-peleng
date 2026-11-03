"""
Peleng Multi-Camera 3D Tracking Pipeline
Universal multi-object tracker with YOLO + BoTSORT + 3D Kalman filtering.
"""

from .camera_manager import (
    CameraManager,
    Camera,
    Intrinsics,
    Extrinsics,
    create_camera_manager
)

from .detector import (
    Detection2D,
    CameraDetections,
    PerCameraTracker,
    MultiCameraDetector,
    create_multi_camera_detector
)

from .triangulator import (
    TriangulationResult,
    Triangulator,
    create_triangulator
)

from .associator import (
    GeometricAssociator,
    AppearanceAssociator,
    HybridAssociator,
    create_associator
)

from .tracker3d import (
    KalmanFilter3D,
    Track3D,
    TrackState,
    MotionModel,
    MultiObjectTracker3D,
    create_tracker3d
)

from .visualizer import (
    Visualizer2D,
    Visualizer3D,
    VisualizerPlots,
    VideoWriter,
    MultiCameraVisualizer,
    create_visualizer
)

from .pipeline import (
    PipelineStats,
    FrameData,
    MultiCameraTrackingPipeline,
    create_pipeline
)

__version__ = "1.0.0"

__all__ = [
    # camera_manager
    "CameraManager",
    "Camera",
    "Intrinsics",
    "Extrinsics",
    "create_camera_manager",
    # detector
    "Detection2D",
    "CameraDetections",
    "PerCameraTracker",
    "MultiCameraDetector",
    "create_multi_camera_detector",
    # triangulator
    "TriangulationResult",
    "Triangulator",
    "create_triangulator",
    # associator
    "GeometricAssociator",
    "AppearanceAssociator",
    "HybridAssociator",
    "create_associator",
    # tracker3d
    "KalmanFilter3D",
    "Track3D",
    "TrackState",
    "MotionModel",
    "MultiObjectTracker3D",
    "create_tracker3d",
    # visualizer
    "Visualizer2D",
    "Visualizer3D",
    "VisualizerPlots",
    "VideoWriter",
    "MultiCameraVisualizer",
    "create_visualizer",
    # pipeline
    "PipelineStats",
    "FrameData",
    "MultiCameraTrackingPipeline",
    "create_pipeline",
]