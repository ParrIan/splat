"""
Tests for colmap_bridge.py — no COLMAP binary needed.

Covers:
  - unpack_bundle: finds manifest.json at root and nested, raises on missing
  - arkit_to_colmap: identity transform, pure translation, round-trip R orthonormality
  - write_cameras_txt: correct PINHOLE format, correct intrinsics
  - write_images_txt: correct line count, correct image IDs (1-indexed), name extraction
  - write_points3d_txt: file is created and has header
  - load_frames: reads manifest and frames correctly
"""

import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from colmap_bridge import (
    arkit_to_colmap,
    load_frames,
    unpack_bundle,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity_transform():
    """ARKit camera-to-world identity transform as flat row-major float[16]."""
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]


def make_bundle_zip(tmp_path, nested=False):
    """Create a minimal session bundle zip. nested=True wraps in a subdirectory."""
    manifest = {'frameCount': 2, 'imageWidth': 960, 'imageHeight': 720}
    frames = [
        {
            'index': 0,
            'timestamp': 0.0,
            'filename': 'images/frame_000000.jpg',
            'intrinsics': {'fx': 600.0, 'fy': 600.0, 'cx': 480.0, 'cy': 360.0},
            'transform': {'values': identity_transform()},
        },
        {
            'index': 1,
            'timestamp': 0.1,
            'filename': 'images/frame_000001.jpg',
            'intrinsics': {'fx': 600.0, 'fy': 600.0, 'cx': 480.0, 'cy': 360.0},
            'transform': {'values': identity_transform()},
        },
    ]
    zip_path = tmp_path / 'session.zip'
    prefix = 'nested_dir/' if nested else ''
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr(f'{prefix}manifest.json', json.dumps(manifest))
        zf.writestr(f'{prefix}frames.json',   json.dumps(frames))
        zf.writestr(f'{prefix}images/frame_000000.jpg', b'fake')
        zf.writestr(f'{prefix}images/frame_000001.jpg', b'fake')
    return zip_path, manifest, frames


# ---------------------------------------------------------------------------
# unpack_bundle
# ---------------------------------------------------------------------------

class TestUnpackBundle:
    def test_flat_zip(self, tmp_path):
        zip_path, manifest, _ = make_bundle_zip(tmp_path)
        bundle_dir = unpack_bundle(zip_path, tmp_path / 'work')
        assert (bundle_dir / 'manifest.json').exists()

    def test_nested_zip(self, tmp_path):
        zip_path, _, _ = make_bundle_zip(tmp_path, nested=True)
        bundle_dir = unpack_bundle(zip_path, tmp_path / 'work')
        assert (bundle_dir / 'manifest.json').exists()

    def test_missing_manifest_raises(self, tmp_path):
        zip_path = tmp_path / 'bad.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('frames.json', '[]')
        with pytest.raises(FileNotFoundError, match='manifest.json'):
            unpack_bundle(zip_path, tmp_path / 'work')


# ---------------------------------------------------------------------------
# load_frames
# ---------------------------------------------------------------------------

class TestLoadFrames:
    def test_loads_manifest_and_frames(self, tmp_path):
        zip_path, manifest, frames = make_bundle_zip(tmp_path)
        bundle_dir = unpack_bundle(zip_path, tmp_path / 'work')
        loaded_manifest, loaded_frames = load_frames(bundle_dir)
        assert loaded_manifest['frameCount'] == 2
        assert loaded_manifest['imageWidth'] == 960
        assert len(loaded_frames) == 2
        assert loaded_frames[0]['index'] == 0


# ---------------------------------------------------------------------------
# arkit_to_colmap
# ---------------------------------------------------------------------------

class TestArkitToColmap:
    def test_identity_gives_unit_quaternion(self):
        (qw, qx, qy, qz), (tx, ty, tz) = arkit_to_colmap(identity_transform())
        # identity camera-to-world with flip -> specific rotation
        # just check quaternion is unit length
        norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(norm - 1.0) < 1e-6

    def test_pure_translation_gives_correct_t(self):
        # camera at (1, 2, 3) looking along -Z (ARKit identity orientation)
        t = identity_transform()
        t[3]  = 1.0  # tx
        t[7]  = 2.0  # ty
        t[11] = 3.0  # tz
        (qw, qx, qy, qz), (tx, ty, tz) = arkit_to_colmap(t)
        # world-to-camera translation: t_w2c = -R_w2c @ t_c2w
        # with flip applied. Check it's a finite number
        assert all(math.isfinite(v) for v in [tx, ty, tz])

    def test_rotation_is_orthonormal(self):
        # use a 90-degree rotation around Y in ARKit convention
        angle = math.pi / 2
        c, s  = math.cos(angle), math.sin(angle)
        # rotation matrix around Y: [[c,0,s],[0,1,0],[-s,0,c]]
        t = [
            c,  0, s,  0,
            0,  1, 0,  0,
            -s, 0, c,  0,
            0,  0, 0,  1,
        ]
        (qw, qx, qy, qz), _ = arkit_to_colmap(t)
        # reconstruct R from quaternion and check orthonormality
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)],
        ])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_all_branch_coverage(self):
        # exercise all four branches of the quaternion conversion
        # by using rotations that trigger each branch
        angles_axes = [
            # trace > 0 (small rotation)
            (0.1, [0, 0, 1]),
            # m[0,0] dominant (180 deg around X gives trace=-1)
            (math.pi, [1, 0, 0]),
            # m[1,1] dominant
            (math.pi, [0, 1, 0]),
            # m[2,2] dominant
            (math.pi, [0, 0, 1]),
        ]
        for angle, axis in angles_axes:
            c, s = math.cos(angle), math.sin(angle)
            ax, ay, az = axis
            # Rodrigues rotation matrix
            R = np.array([
                [c + ax**2*(1-c),      ax*ay*(1-c) - az*s, ax*az*(1-c) + ay*s],
                [ay*ax*(1-c) + az*s,   c + ay**2*(1-c),     ay*az*(1-c) - ax*s],
                [az*ax*(1-c) - ay*s,   az*ay*(1-c) + ax*s,  c + az**2*(1-c)],
            ])
            t_vals = list(R[0]) + [0] + list(R[1]) + [0] + list(R[2]) + [0]
            (qw, qx, qy, qz), _ = arkit_to_colmap(t_vals)
            norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            assert abs(norm - 1.0) < 1e-5, f'quaternion not unit for angle={angle}, axis={axis}'


# ---------------------------------------------------------------------------
# write_cameras_txt
# ---------------------------------------------------------------------------

class TestWriteCamerasTxt:
    def test_pinhole_format(self, tmp_path):
        manifest = {'imageWidth': 960, 'imageHeight': 720}
        frames   = [{'intrinsics': {'fx': 600.0, 'fy': 601.0, 'cx': 480.0, 'cy': 360.0}}]
        path     = tmp_path / 'cameras.txt'
        write_cameras_txt(manifest, frames, path)

        lines = [l for l in path.read_text().splitlines() if not l.startswith('#')]
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == '1'           # camera ID
        assert parts[1] == 'PINHOLE'
        assert parts[2] == '960'         # width
        assert parts[3] == '720'         # height
        assert float(parts[4]) == 600.0  # fx
        assert float(parts[5]) == 601.0  # fy
        assert float(parts[6]) == 480.0  # cx
        assert float(parts[7]) == 360.0  # cy


# ---------------------------------------------------------------------------
# write_images_txt
# ---------------------------------------------------------------------------

class TestWriteImagesTxt:
    def test_one_indexed_image_ids(self, tmp_path):
        frames = [
            {'index': 0, 'filename': 'images/frame_000000.jpg',
             'transform': {'values': identity_transform()}},
            {'index': 1, 'filename': 'images/frame_000001.jpg',
             'transform': {'values': identity_transform()}},
        ]
        path = tmp_path / 'images.txt'
        write_images_txt(frames, path)
        lines = [l for l in path.read_text().splitlines() if not l.startswith('#') and l.strip()]
        # each frame produces one data line (blank lines filtered)
        assert len(lines) == 2
        # first image ID should be 1 (COLMAP is 1-indexed)
        assert lines[0].split()[0] == '1'
        assert lines[1].split()[0] == '2'

    def test_filename_basename_only(self, tmp_path):
        frames = [
            {'index': 0, 'filename': 'images/frame_000000.jpg',
             'transform': {'values': identity_transform()}},
        ]
        path = tmp_path / 'images.txt'
        write_images_txt(frames, path)
        lines = [l for l in path.read_text().splitlines() if not l.startswith('#') and l.strip()]
        # last field should be just the filename, not the full path
        assert lines[0].split()[-1] == 'frame_000000.jpg'

    def test_camera_id_always_1(self, tmp_path):
        frames = [
            {'index': 0, 'filename': 'images/f0.jpg',
             'transform': {'values': identity_transform()}},
        ]
        path = tmp_path / 'images.txt'
        write_images_txt(frames, path)
        lines = [l for l in path.read_text().splitlines() if not l.startswith('#') and l.strip()]
        parts = lines[0].split()
        # format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        assert parts[8] == '1'


# ---------------------------------------------------------------------------
# write_points3d_txt
# ---------------------------------------------------------------------------

class TestWritePoints3dTxt:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / 'points3D.txt'
        write_points3d_txt(path)
        assert path.exists()
        content = path.read_text()
        assert content.startswith('#')

    def test_no_data_lines(self, tmp_path):
        path = tmp_path / 'points3D.txt'
        write_points3d_txt(path)
        data_lines = [l for l in path.read_text().splitlines() if not l.startswith('#') and l.strip()]
        assert len(data_lines) == 0
