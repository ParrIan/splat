"""
visualize_sparse.py

Visualizes COLMAP sparse reconstruction — point cloud + camera frustums.

Usage:
    python visualize_sparse.py <sparse_dir>
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import open3d as o3d


def read_points3d_bin(path: Path):
    points = []
    colors = []

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)  # skip track data (image_id, point2d_idx pairs)
            points.append([x, y, z])
            colors.append([r / 255, g / 255, b / 255])

    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.float64)


def read_images_bin(path: Path):
    """Returns list of camera-to-world 4x4 transforms."""
    transforms = []

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            _camera_id = struct.unpack("<I", f.read(4))[0]

            # read null-terminated name
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c

            n_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n_points2d)  # skip 2D observations

            # quaternion to rotation matrix (world-to-camera)
            R_w2c = quat_to_rotation(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])

            # camera-to-world
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t

            T = np.eye(4)
            T[:3, :3] = R_c2w
            T[:3, 3] = t_c2w
            transforms.append(T)

    return transforms


def quat_to_rotation(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def make_camera_frustum(T: np.ndarray, scale: float = 0.05):
    """Creates a small frustum wireframe at the given camera-to-world transform."""
    # frustum corners in camera space — pointing along -Z (into scene, COLMAP convention)
    w, h, f = 0.8, 0.6, 1.0
    corners_cam = np.array([
        [0, 0, 0],
        [ w*f,  h*f, -f],
        [-w*f,  h*f, -f],
        [-w*f, -h*f, -f],
        [ w*f, -h*f, -f],
    ]) * scale

    corners_world = (T[:3, :3] @ corners_cam.T).T + T[:3, 3]

    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    colors = [[1, 0.5, 0]] * len(lines)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sparse_dir", type=Path)
    parser.add_argument("--max-cameras", type=int, default=100,
                        help="max camera frustums to draw (every Nth camera)")
    args = parser.parse_args()

    sparse_dir = args.sparse_dir.resolve()
    points_bin = sparse_dir / "points3D.bin"
    images_bin = sparse_dir / "images.bin"

    print(f"loading point cloud from {points_bin.name}...")
    points, colors = read_points3d_bin(points_bin)
    print(f"  {len(points):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    if images_bin.exists():
        print(f"loading camera poses from {images_bin.name}...")
        transforms = read_images_bin(images_bin)
        print(f"  {len(transforms)} cameras")

        step = max(1, len(transforms) // args.max_cameras)
        frustums = [make_camera_frustum(T) for T in transforms[::step]]
        geometries.extend(frustums)
        print(f"  drawing every {step}th camera ({len(frustums)} frustums)")

    print("\ncontrols: left drag = rotate, right drag = pan, scroll = zoom, q = quit")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Sparse Reconstruction",
        width=1280,
        height=800,
    )


if __name__ == "__main__":
    main()
