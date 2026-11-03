import csv
import json
import math
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAJECTORY_DIR = RUNS_DIR / "trajectory"
VALIDATION_DIR = RUNS_DIR / "validation"

TRACK_FILE = DATA_DIR / "object_transform_SM_AA_track.csv"


def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="cp1251", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            frame = int(row["Frame"])
            gt[frame] = {
                "x": float(row["Loc_X"]),
                "y": float(row["Loc_Y"]),
                "z": float(row["Loc_Z"]),
            }
    return gt


def load_trajectory(path):
    traj = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            if row["x"] == "" or row["x"] is None:
                continue
            traj[frame] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
    return traj


def validate(trajectory_path, ground_truth_path, output_path=None):
    gt = load_ground_truth(ground_truth_path)
    traj = load_trajectory(trajectory_path)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = VALIDATION_DIR / "validation_report.csv"

    common_frames = sorted(set(gt.keys()) & set(traj.keys()))
    gt_only = sorted(set(gt.keys()) - set(traj.keys()))
    traj_only = sorted(set(traj.keys()) - set(gt.keys()))

    errors = []
    for frame in common_frames:
        g = gt[frame]
        t = traj[frame]
        dx = t["x"] - g["x"]
        dy = t["y"] - g["y"]
        dz = t["z"] - g["z"]
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        errors.append({
            "frame": frame,
            "gt_x": g["x"], "gt_y": g["y"], "gt_z": g["z"],
            "est_x": t["x"], "est_y": t["y"], "est_z": t["z"],
            "dx": dx, "dy": dy, "dz": dz,
            "error_m": dist,
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "gt_x", "gt_y", "gt_z",
            "est_x", "est_y", "est_z",
            "dx", "dy", "dz", "error_m",
        ])
        for e in errors:
            writer.writerow([
                e["frame"], e["gt_x"], e["gt_y"], e["gt_z"],
                e["est_x"], e["est_y"], e["est_z"],
                e["dx"], e["dy"], e["dz"], e["error_m"],
            ])

    if errors:
        errs = [e["error_m"] for e in errors]
        mean_err = sum(errs) / len(errs)
        max_err = max(errs)
        rmse = math.sqrt(sum(e ** 2 for e in errs) / len(errs))

        # Per-axis errors
        dxs = [abs(e["dx"]) for e in errors]
        dys = [abs(e["dy"]) for e in errors]
        dzs = [abs(e["dz"]) for e in errors]

        print(f"\n=== Validation Report ===")
        print(f"Common frames: {len(common_frames)}")
        print(f"GT only (not reconstructed): {len(gt_only)}")
        print(f"Traj only (not in GT): {len(traj_only)}")
        print(f"Mean error: {mean_err:.2f} m")
        print(f"RMSE: {rmse:.2f} m")
        print(f"Max error: {max_err:.2f} m")
        print(f"Mean |dx|: {sum(dxs)/len(dxs):.2f} m")
        print(f"Mean |dy|: {sum(dys)/len(dys):.2f} m")
        print(f"Mean |dz|: {sum(dzs)/len(dzs):.2f} m")

        # Save summary
        summary_path = VALIDATION_DIR / "validation_summary.json"
        summary = {
            "common_frames": len(common_frames),
            "gt_only": len(gt_only),
            "traj_only": len(traj_only),
            "mean_error_m": mean_err,
            "rmse_m": rmse,
            "max_error_m": max_err,
            "mean_abs_dx_m": sum(dxs) / len(dxs),
            "mean_abs_dy_m": sum(dys) / len(dys),
            "mean_abs_dz_m": sum(dzs) / len(dzs),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport: {output_path}")
        print(f"Summary: {summary_path}")
    else:
        print("No common frames for validation.")

    return errors


if __name__ == "__main__":
    traj_path = TRAJECTORY_DIR / "trajectory_3d.csv"
    if not traj_path.exists():
        print(f"Trajectory not found: {traj_path}. Run triangulate.py first.")
    else:
        validate(traj_path, TRACK_FILE)
