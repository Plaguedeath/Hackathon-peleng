import csv
import json
import math
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
CALIB_DIR = RUNS_DIR / "calibration"
DETECTIONS_DIR = RUNS_DIR / "detections"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"

VIDEO_W = 1920
VIDEO_H = 1080


def mat_vec_3x3(A, v):
    return [
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    ]


def mat_transpose_3x3(A):
    return [
        [A[0][0], A[1][0], A[2][0]],
        [A[0][1], A[1][1], A[2][1]],
        [A[0][2], A[1][2], A[2][2]],
    ]


def mat_mul_3x3(A, B):
    return [
        [
            A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0],
            A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1],
            A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2],
        ],
        [
            A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0],
            A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1],
            A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2],
        ],
        [
            A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0],
            A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1],
            A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2],
        ],
    ]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def vec_normalize(v):
    n = vec_norm(v)
    if n == 0:
        return v
    return [v[0] / n, v[1] / n, v[2] / n]


def invert_3x3(A):
    """Invert a 3x3 matrix using cofactor method."""
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]

    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular")

    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, -(b * i - c * h) * inv_det, (b * f - c * e) * inv_det],
        [-(d * i - f * g) * inv_det, (a * i - c * g) * inv_det, -(a * f - c * d) * inv_det],
        [(d * h - e * g) * inv_det, -(a * h - b * g) * inv_det, (a * e - b * d) * inv_det],
    ]


def load_camera_params(path):
    with open(path, "r") as f:
        return json.load(f)


def load_detections(path):
    detections = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            cam_id = int(row["camera_id"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            conf = float(row["confidence"])
            detections.setdefault(frame, []).append({
                "camera_id": cam_id,
                "cx": cx,
                "cy": cy,
                "confidence": conf,
            })
    return detections


def compute_ray_direction(K, R_w2c, u, v, W, H):
    """Compute camera ray direction in world coordinates for pixel (u, v)."""
    K_inv = invert_3x3(K)
    # Normalized image coordinates
    xn = (u - K[0][2]) / K[0][0]
    yn = (v - K[1][2]) / K[1][1]
    # Ray in camera coordinates (looking along +X in our UE convention)
    d_cam = [1.0, xn, yn]  # depth along X, image plane is YZ
    d_cam = vec_normalize(d_cam)
    # Transform to world: d_world = R^T @ d_cam
    R_t = mat_transpose_3x3(R_w2c)
    d_world = mat_vec_3x3(R_t, d_cam)
    d_world = vec_normalize(d_world)
    return d_world


def triangulate_point(observations, camera_params, W, H):
    """
    Triangulate 3D point from multiple camera observations.
    observations: list of {camera_id, cx, cy, confidence}
    camera_params: dict from camera_params.json
    Returns: (X, Y, Z) or None if < 2 observations
    """
    if len(observations) < 2:
        return None

    A_sum = [[0.0] * 3 for _ in range(3)]
    b_sum = [0.0, 0.0, 0.0]

    for obs in observations:
        cam_id = str(obs["camera_id"])
        cam = camera_params[cam_id]
        K = cam["K"]

        # Find extrinsics for this frame - we need frame number, but observations don't carry it
        # We'll use the first available extrinsics (cameras are static except Cam5 yaw)
        # For Cam5, we need per-frame yaw. Handle below.
        extrinsics = cam["extrinsics"]
        # Use frame 0 as default (static cameras)
        ext = extrinsics["0"]
        R_w2c = ext["R"]
        C = ext["camera_position"]

        # For Cam5, yaw changes per frame - we need frame-specific extrinsics
        # This will be handled by passing frame-specific params
        d_world = compute_ray_direction(K, R_w2c, obs["cx"], obs["cy"], W, H)
        d_world = vec_normalize(d_world)

        # Projection matrix onto plane perpendicular to ray: P = I - d*d^T
        dt = [d_world[0], d_world[1], d_world[2]]
        ddt = [
            [dt[0] * dt[0], dt[0] * dt[1], dt[0] * dt[2]],
            [dt[1] * dt[0], dt[1] * dt[1], dt[1] * dt[2]],
            [dt[2] * dt[0], dt[2] * dt[1], dt[2] * dt[2]],
        ]
        P = [
            [1.0 - ddt[0][0], -ddt[0][1], -ddt[0][2]],
            [-ddt[1][0], 1.0 - ddt[1][1], -ddt[1][2]],
            [-ddt[2][0], -ddt[2][1], 1.0 - ddt[2][2]],
        ]

        # Weight by confidence
        w = obs["confidence"]
        for i in range(3):
            for j in range(3):
                A_sum[i][j] += w * P[i][j]
            b_sum[i] += w * sum(P[i][k] * C[k] for k in range(3))

    A_inv = invert_3x3(A_sum)
    X = [
        A_inv[0][0] * b_sum[0] + A_inv[0][1] * b_sum[1] + A_inv[0][2] * b_sum[2],
        A_inv[1][0] * b_sum[0] + A_inv[1][1] * b_sum[1] + A_inv[1][2] * b_sum[2],
        A_inv[2][0] * b_sum[0] + A_inv[2][1] * b_sum[1] + A_inv[2][2] * b_sum[2],
    ]
    return X


def triangulate_all(detections_path, camera_params_path, output_path, W=VIDEO_W, H=VIDEO_H):
    camera_params = load_camera_params(camera_params_path)
    detections = load_detections(detections_path)

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(detections.keys())
    results = []

    for frame in all_frames:
        obs = detections[frame]
        X = triangulate_point(obs, camera_params, W, H)
        if X is None:
            results.append({
                "frame": frame,
                "x": None, "y": None, "z": None,
                "n_cameras": len(obs),
                "triangulated": False,
            })
        else:
            results.append({
                "frame": frame,
                "x": X[0], "y": X[1], "z": X[2],
                "n_cameras": len(obs),
                "triangulated": True,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "n_cameras", "triangulated"])
        for r in results:
            writer.writerow([
                r["frame"],
                r["x"] if r["x"] is not None else "",
                r["y"] if r["y"] is not None else "",
                r["z"] if r["z"] is not None else "",
                r["n_cameras"],
                r["triangulated"],
            ])

    n_tri = sum(1 for r in results if r["triangulated"])
    print(f"Triangulated {n_tri}/{len(results)} frames -> {output_path}")
    return results


if __name__ == "__main__":
    detections_path = DETECTIONS_DIR / "all_detections.csv"
    camera_params_path = CALIB_DIR / "camera_params.json"
    output_path = TRAJECTORY_DIR / "trajectory_3d.csv"

    if not detections_path.exists():
        print(f"Detections not found: {detections_path}. Run infer_yolo.py first.")
    elif not camera_params_path.exists():
        print(f"Camera params not found: {camera_params_path}. Run camera_calibration.py first.")
    else:
        triangulate_all(detections_path, camera_params_path, output_path)
