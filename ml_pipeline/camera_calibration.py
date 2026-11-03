import csv
import json
import math
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "runs" / "calibration"

SENSOR_W_MM = 23.76
SENSOR_H_MM = 13.365

CAMERA_FILES = {
    1: DATA_DIR / "object_transform_CineCameraActor1.csv",
    2: DATA_DIR / "object_transform_CineCameraActor2.csv",
    3: DATA_DIR / "object_transform_CineCameraActor3.csv",
    4: DATA_DIR / "object_transform_CineCameraActor4.csv",
    5: DATA_DIR / "object_transform_CineCameraActor5.csv",
}

TRACK_FILE = DATA_DIR / "object_transform_SM_AA_track.csv"


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


def euler_to_rotation(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]

    Ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]

    Rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return mat_mul_3x3(mat_mul_3x3(Rz, Ry), Rx)


def compute_K(focal_mm, W_px, H_px):
    fx = focal_mm * W_px / SENSOR_W_MM
    fy = focal_mm * H_px / SENSOR_H_MM
    cx = W_px / 2.0
    cy = H_px / 2.0
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def project_point(K, R_w2c, t_w2c, X_world):
    X_cam = mat_vec_3x3(R_w2c, X_world)
    X_cam = [X_cam[0] + t_w2c[0], X_cam[1] + t_w2c[1], X_cam[2] + t_w2c[2]]
    depth = X_cam[0]
    if depth <= 0:
        return None, X_cam
    u = K[0][0] * X_cam[1] / depth + K[0][2]
    v = K[1][1] * X_cam[2] / depth + K[1][2]
    return (u, v), X_cam


def read_camera_csv(path):
    with open(path, "r", encoding="cp1251", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    header = rows[0]
    focal_length = None
    for i, cell in enumerate(header):
        if cell.strip() == "Focal length" and i + 1 < len(header):
            try:
                focal_length = float(header[i + 1])
            except ValueError:
                pass

    frames = []
    for row in rows[1:]:
        if len(row) < 7:
            continue
        try:
            frame = int(row[0])
            loc_x = float(row[1])
            loc_y = float(row[2])
            loc_z = float(row[3])
            rot_roll = float(row[4])
            rot_pitch = float(row[5])
            rot_yaw = float(row[6])
        except ValueError:
            continue
        frames.append({
            "frame": frame,
            "loc_x": loc_x,
            "loc_y": loc_y,
            "loc_z": loc_z,
            "rot_roll": rot_roll,
            "rot_pitch": rot_pitch,
            "rot_yaw": rot_yaw,
        })

    return focal_length, frames


def read_track_csv(path):
    with open(path, "r", encoding="cp1251", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    points = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        try:
            frame = int(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
        except ValueError:
            continue
        points.append({
            "frame": frame,
            "x": x,
            "y": y,
            "z": z,
        })

    return points


def main():
    W_px = int(os.environ.get("VIDEO_W", "1920"))
    H_px = int(os.environ.get("VIDEO_H", "1080"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cameras = {}
    for cam_id, path in CAMERA_FILES.items():
        focal_length, frames = read_camera_csv(path)
        K = compute_K(focal_length, W_px, H_px)
        cameras[cam_id] = {
            "focal_length_mm": focal_length,
            "K": K,
            "frames": frames,
        }

    track = read_track_csv(TRACK_FILE)
    track_by_frame = {pt["frame"]: pt for pt in track}

    all_frames = set(track_by_frame.keys())
    for cam in cameras.values():
        for fr in cam["frames"]:
            all_frames.add(fr["frame"])
    all_frames = sorted(all_frames)

    projections = []
    camera_params = {}

    for cam_id, cam in cameras.items():
        cam_params = {
            "focal_length_mm": cam["focal_length_mm"],
            "K": cam["K"],
            "extrinsics": {},
        }
        for fr in cam["frames"]:
            frame = fr["frame"]
            R = euler_to_rotation(fr["rot_roll"], fr["rot_pitch"], fr["rot_yaw"])
            C = [fr["loc_x"], fr["loc_y"], fr["loc_z"]]
            Rt = mat_transpose_3x3(R)
            t = mat_vec_3x3(Rt, [-C[0], -C[1], -C[2]])
            cam_params["extrinsics"][str(frame)] = {
                "R": R,
                "t": t,
                "camera_position": C,
            }

        camera_params[str(cam_id)] = cam_params

    for frame in all_frames:
        if frame not in track_by_frame:
            continue
        pt = track_by_frame[frame]
        X_world = [pt["x"], pt["y"], pt["z"]]

        for cam_id, cam in cameras.items():
            fr_map = {f["frame"]: f for f in cam["frames"]}
            if frame not in fr_map:
                continue
            fr = fr_map[frame]
            R = euler_to_rotation(fr["rot_roll"], fr["rot_pitch"], fr["rot_yaw"])
            C = [fr["loc_x"], fr["loc_y"], fr["loc_z"]]
            Rt = mat_transpose_3x3(R)
            t = mat_vec_3x3(Rt, [-C[0], -C[1], -C[2]])
            K = cam["K"]

            (uv, X_cam) = project_point(K, Rt, t, X_world)

            entry = {
                "frame": frame,
                "camera_id": cam_id,
                "object_x": pt["x"],
                "object_y": pt["y"],
                "object_z": pt["z"],
                "u": uv[0] if uv else None,
                "v": uv[1] if uv else None,
                "depth": X_cam[2] if X_cam is not None else None,
                "visible": uv is not None and 0 <= uv[0] < W_px and 0 <= uv[1] < H_px and X_cam[0] > 0,
            }
            projections.append(entry)

    camera_params_serializable = {}
    for cam_id, cam in camera_params.items():
        camera_params_serializable[cam_id] = {
            "focal_length_mm": cam["focal_length_mm"],
            "K": cam["K"],
            "extrinsics": {},
        }
        for frame_str, ext in cam["extrinsics"].items():
            camera_params_serializable[cam_id]["extrinsics"][frame_str] = {
                "R": ext["R"],
                "t": ext["t"],
                "camera_position": ext["camera_position"],
            }

    with open(OUT_DIR / "camera_params.json", "w") as f:
        json.dump(camera_params_serializable, f, indent=2)

    with open(OUT_DIR / "projections.json", "w") as f:
        json.dump(projections, f, indent=2)

    csv_path = OUT_DIR / "projections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_id", "object_x", "object_y", "object_z", "u", "v", "depth", "visible"])
        for p in projections:
            writer.writerow([
                p["frame"], p["camera_id"],
                p["object_x"], p["object_y"], p["object_z"],
                p["u"], p["v"], p["depth"], p["visible"],
            ])

    visible_count = sum(1 for p in projections if p["visible"])
    total_count = len(projections)

    print("Resolution: {}x{}".format(W_px, H_px))
    print("Cameras calibrated: {}".format(len(cameras)))
    print("Total projections: {}".format(total_count))
    print("Visible projections: {}".format(visible_count))
    print("Outputs written to: {}".format(OUT_DIR))
    print("  - camera_params.json")
    print("  - projections.json")
    print("  - projections.csv")


if __name__ == "__main__":
    main()
