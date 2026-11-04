#!/usr/bin/env python
"""
Collect per-frame detection and collision data for visualization.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import create_pipeline
from src.camera_manager import create_camera_manager
from src.tracker3d import TrackState


def collect_per_frame_data(max_frames=200):
    """Run pipeline and collect per-frame statistics."""
    print("Loading pipeline...")
    pipeline = create_pipeline("config/tracker.yaml")
    pipeline.initialize()

    # Data collectors
    frame_data_list = []

    # Track proximity threshold (cm) - consider "collision" if tracks closer than this
    COLLISION_THRESHOLD_CM = 500  # 5 meters

    def progress_callback(frame_idx, total_frames, stats):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{total_frames}...")

    # We need to hook into the pipeline to get per-frame data
    # Let's run sync mode and collect data at each frame
    pipeline.running = True
    pipeline.start_time = 0
    pipeline.frame_idx = 0

    total_frames = pipeline._estimate_total_frames()
    total_frames = min(total_frames, max_frames)

    print(f"Processing {total_frames} frames...")

    for frame_idx in range(total_frames):
        # Read frames
        frames = pipeline.camera_manager.read_all()
        if not any(f is not None for f in frames.values()):
            break

        # Create FrameData
        from src.pipeline import FrameData
        frame_data = FrameData(
            frame_idx=frame_idx,
            timestamp=frame_idx / 30.0,
            frames=frames,
            detections={}
        )

        # Process through all stages
        for stage in pipeline.stages:
            frame_data = stage.process(frame_data)

        # Collect statistics
        detections_2d = sum(len(d.detections) for d in frame_data.detections.values())
        detections_per_cam = {cam_id: len(d.detections) for cam_id, d in frame_data.detections.items()}

        # Get 3D tracks
        tracks_3d = frame_data.tracks_3d or {}
        active_tracks = {tid: t for tid, t in tracks_3d.items() if t.is_active()}

        # Check for track proximity ("collisions")
        collisions = []
        track_ids = list(active_tracks.keys())
        for i, tid1 in enumerate(track_ids):
            for tid2 in track_ids[i+1:]:
                pos1 = active_tracks[tid1].get_position()
                pos2 = active_tracks[tid2].get_position()
                if pos1 is not None and pos2 is not None:
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < COLLISION_THRESHOLD_CM:
                        collisions.append({
                            'track1': tid1,
                            'track2': tid2,
                            'distance_cm': dist
                        })

        # Track 3D positions for trajectory
        track_positions = {}
        for tid, track in active_tracks.items():
            pos = track.get_position()
            if pos is not None:
                track_positions[tid] = pos.tolist()

        frame_data_list.append({
            'frame_idx': frame_idx,
            'timestamp': frame_idx / 30.0,
            'total_detections_2d': detections_2d,
            'detections_per_cam': detections_per_cam,
            'num_tracks_3d': len(active_tracks),
            'track_ids': list(active_tracks.keys()),
            'track_positions': track_positions,
            'collisions': collisions,
            'num_collisions': len(collisions)
        })

        if frame_idx % 20 == 0:
            progress_callback(frame_idx, total_frames, None)

    pipeline.stop()
    pipeline._finalize()

    return frame_data_list


def save_frame_data(frame_data_list, output_path="output/frame_data.npy"):
    """Save collected frame data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, frame_data_list, allow_pickle=True)
    print(f"Saved frame data to {output_path}")


def load_frame_data(input_path="output/frame_data.npy"):
    """Load collected frame data."""
    return np.load(input_path, allow_pickle=True).tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--output", type=str, default="output/frame_data.npy")
    args = parser.parse_args()

    data = collect_per_frame_data(args.max_frames)
    save_frame_data(data, args.output)

    # Print summary
    print(f"\nCollected {len(data)} frames")
    print(f"Total 2D detections: {sum(d['total_detections_2d'] for d in data)}")
    print(f"Max 3D tracks: {max(d['num_tracks_3d'] for d in data)}")
    print(f"Total collisions: {sum(d['num_collisions'] for d in data)}")
    print(f"Frames with collisions: {sum(1 for d in data if d['num_collisions'] > 0)}")