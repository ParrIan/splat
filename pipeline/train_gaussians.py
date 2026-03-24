"""
Train 3D Gaussian Splatting on a COLMAP scene.

Requires the gaussian-splatting repo installed at /content/gaussian-splatting.

Usage (Colab):
    python pipeline/train_gaussians.py \
        --zip  /drive/MyDrive/splat/desk_colmap.zip \
        --out  /drive/MyDrive/splat \
        --stem desk_colmap

Outputs:
    <out>/<stem>_gaussian_<iterations>.ply
"""

import argparse
import glob
import os
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path

WORK_DIR   = Path('/content/splat_work')
GS_REPO    = Path('/content/gaussian-splatting')


def unpack_zip(zip_path, work_dir):
    scene_dir = work_dir / 'scene'
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(scene_dir)
    candidates = glob.glob(str(scene_dir / '**' / 'sparse'), recursive=True)
    if candidates:
        scene_dir = Path(candidates[0]).parent
    print(f'scene dir: {scene_dir}')
    return scene_dir


def check_scene(scene_dir):
    sparse_dir = scene_dir / 'sparse' / '0'
    with open(sparse_dir / 'points3D.bin', 'rb') as f:
        n_pts = struct.unpack('<Q', f.read(8))[0]
    with open(sparse_dir / 'images.bin', 'rb') as f:
        n_cams = struct.unpack('<Q', f.read(8))[0]
    print(f'sparse points: {n_pts:,}')
    print(f'registered cameras: {n_cams:,}')
    if n_pts < 100:
        raise RuntimeError(f'too few sparse points ({n_pts}) — reconstruction may have failed')


def train(scene_dir, output_dir, iterations):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python', str(GS_REPO / 'train.py'),
        '-s', str(scene_dir),
        '-m', str(output_dir),
        '--iterations', str(iterations),
        '--test_iterations', str(iterations),
        '--save_iterations', str(iterations),
    ]
    print(f'training {iterations} iterations...')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError('training failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip',        required=True, help='colmap scene zip')
    parser.add_argument('--out',        required=True, help='output directory (Drive)')
    parser.add_argument('--stem',       required=True, help='output filename stem')
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--sanity',     type=int, default=0,
                        help='if >0, run a quick sanity check at this iteration count first')
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dir  = unpack_zip(args.zip, WORK_DIR)
    output_dir = WORK_DIR / 'gs_output'

    check_scene(scene_dir)

    if args.sanity > 0:
        print(f'\n--- sanity check at {args.sanity} iterations ---')
        train(scene_dir, output_dir, args.sanity)
        ply_src = output_dir / f'point_cloud/iteration_{args.sanity}/point_cloud.ply'
        print(f'sanity PLY: {ply_src} ({ply_src.stat().st_size / 1024 / 1024:.1f} MB)')

    print(f'\n--- full training at {args.iterations} iterations ---')
    train(scene_dir, output_dir, args.iterations)

    ply_src = output_dir / f'point_cloud/iteration_{args.iterations}/point_cloud.ply'
    ply_dst = out_dir / f'{args.stem}_gaussian_{args.iterations}.ply'
    shutil.copy(ply_src, ply_dst)
    print(f'\nsaved: {ply_dst}')


if __name__ == '__main__':
    main()
