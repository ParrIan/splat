"""
colmap_bridge.py

Converts a splat session bundle into COLMAP sparse reconstruction input,
runs feature extraction and point triangulation using ARKit poses directly.

Usage:
    python colmap_bridge.py <session_zip> [--output <dir>] [--colmap <path>]

Output structure:
    <output>/
        images/          # symlinked or copied from bundle
        sparse/0/
            cameras.txt
            images.txt
            points3D.txt # empty initially, filled by triangulator
"""

import argparse
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np


def unpack_bundle(zip_path: Path, work_dir: Path) -> Path:
    bundle_dir = work_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(bundle_dir)

    # find the actual root — handles zips with a nested top-level directory
    manifest_candidates = list(bundle_dir.rglob("manifest.json"))
    if not manifest_candidates:
        raise FileNotFoundError(f"manifest.json not found in {zip_path.name}")
    return manifest_candidates[0].parent


def load_frames(bundle_dir: Path) -> tuple[dict, list[dict]]:

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    frames = json.loads((bundle_dir / "frames.json").read_text())
    return manifest, frames


def arkit_to_colmap(transform_values: list[float]):
    """
    Convert ARKit camera-to-world transform to COLMAP world-to-camera (q, t).

    ARKit convention: camera looks along -Z, Y up, right-handed, camera-to-world.
    COLMAP convention: camera looks along +Z, Y down, right-handed, world-to-camera.

    The coordinate system change is a 180 degree rotation around X:
        flip = diag(1, -1, -1)
    Applied as: T_colmap_c2w = flip @ T_arkit_c2w
    Then invert to get world-to-camera for COLMAP.
    """
    r = transform_values
    T_c2w = np.array([
        [r[0],  r[1],  r[2],  r[3]],
        [r[4],  r[5],  r[6],  r[7]],
        [r[8],  r[9],  r[10], r[11]],
        [0,     0,     0,     1   ],
    ], dtype=np.float64)

    # flip Y and Z axes to go from ARKit to COLMAP camera convention
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    T_c2w_col = flip @ T_c2w

    # invert to get world-to-camera
    R_c2w = T_c2w_col[:3, :3]
    t_c2w = T_c2w_col[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    # rotation matrix to quaternion
    m = R_w2c
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    return (qw, qx, qy, qz), (t_w2c[0], t_w2c[1], t_w2c[2])


def write_cameras_txt(manifest: dict, frames: list[dict], path: Path):
    """
    Single shared camera model (PINHOLE) — all frames use the same intrinsics.
    Uses intrinsics from the first frame; ARKit intrinsics are stable across frames
    for a fixed capture resolution.
    """
    intr = frames[0]["intrinsics"]
    w = manifest["imageWidth"]
    h = manifest["imageHeight"]
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")


def write_images_txt(frames: list[dict], path: Path):
    """
    One entry per frame. COLMAP image format:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    (followed by blank line — no 2D point observations yet, triangulator fills these)
    """
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for frame in frames:
            image_id = frame["index"] + 1  # COLMAP is 1-indexed
            t = frame["transform"]["values"]
            (qw, qx, qy, qz), (tx, ty, tz) = arkit_to_colmap(t)
            name = Path(frame["filename"]).name

            f.write(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {name}\n")
            f.write("\n")


def write_points3d_txt(path: Path):
    # empty — triangulator will populate this
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


def run_colmap(colmap_bin: str, image_dir: Path, sparse_dir: Path, database_path: Path, sfm: bool = False):
    def run(args):
        print(f"  running: {' '.join(args)}")
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"COLMAP failed: {args[0]}")

    # feature extraction
    run([
        colmap_bin, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
    ])

    # sequential matching — overlap=50 gives wide baseline
    run([
        colmap_bin, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "50",
    ])

    if sfm:
        # full SfM — COLMAP estimates poses itself, ignores ARKit
        run([
            colmap_bin, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir.parent),
        ])
    else:
        # triangulate using known ARKit poses
        run([
            colmap_bin, "point_triangulator",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--input_path", str(sparse_dir),
            "--output_path", str(sparse_dir),
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_zip", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--colmap", type=str, default="colmap")
    parser.add_argument("--arkit", action="store_true", help="use ARKit poses instead of COLMAP full SfM")
    args = parser.parse_args()

    zip_path = args.session_zip.resolve()
    output_dir = (args.output or zip_path.parent / zip_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"unpacking {zip_path.name}...")
    bundle_dir = unpack_bundle(zip_path, output_dir / "work")
    manifest, frames = load_frames(bundle_dir)
    print(f"  {manifest['frameCount']} frames, {manifest['imageWidth']}x{manifest['imageHeight']}")

    # copy images
    image_dir = output_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    shutil.copytree(bundle_dir / "images", image_dir)

    # write COLMAP sparse input
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"

    if args.arkit:
        print("mode: ARKit poses + triangulation")
        print("writing COLMAP cameras.txt...")
        write_cameras_txt(manifest, frames, sparse_dir / "cameras.txt")
        print("writing COLMAP images.txt...")
        write_images_txt(frames, sparse_dir / "images.txt")
        write_points3d_txt(sparse_dir / "points3D.txt")
    else:
        print("mode: full SfM (COLMAP estimates poses)")

    print("running COLMAP...")
    run_colmap(args.colmap, image_dir, sparse_dir, database_path, sfm=not args.arkit)

    points_bin = sparse_dir / "points3D.bin"
    if points_bin.exists() and points_bin.stat().st_size > 64:
        print(f"\ndone. sparse reconstruction at:\n  {sparse_dir}")
        print(f"  points3D.bin: {points_bin.stat().st_size / 1024:.0f} KB")
    else:
        print("\nwarning: points3D.bin is empty or missing — triangulation may have failed")

    print("\nnext step:")
    print(f"  python train.py --colmap {output_dir}")


if __name__ == "__main__":
    main()
