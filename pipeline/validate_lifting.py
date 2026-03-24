"""
Validate mask-to-Gaussian lifting output.

Checks:
  1. Assignment coverage — what fraction of Gaussians are assigned
  2. Spatial compactness — std of positions within each cluster
  3. Bbox separation — no two object nodes have excessive bbox overlap
  4. Visualizes Gaussian clusters in open3d, color-coded by object

Usage:
    python pipeline/validate_lifting.py \
        --gaussians desk_colmap_gaussian_15000.ply \
        --assignments pipeline/assignments.json

Input (assignments.json schema):
    {
      "<gaussian_idx>": "<object_id>"
    }

Input (objects.json schema):
    {
      "<object_id>": {
        "label": str,
        "type": "object" | "structure" | "background",
        "gaussian_ids": [int, ...],
        "bbox": {
          "min": [x, y, z],
          "max": [x, y, z],
          "centroid": [x, y, z]
        }
      }
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

COMPACTNESS_WARN  = 0.5   # warn if cluster std > this (meters)
OVERLAP_WARN      = 0.3   # warn if two object bbox IoU > this
MIN_COVERAGE      = 0.5   # warn if <50% of gaussians assigned to non-background


def load_gaussians(ply_path):
    ply = PlyData.read(str(ply_path))
    v   = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    return xyz


def bbox_iou_3d(bbox_a, bbox_b):
    min_a, max_a = np.array(bbox_a['min']), np.array(bbox_a['max'])
    min_b, max_b = np.array(bbox_b['min']), np.array(bbox_b['max'])
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol  = inter_dims.prod()
    vol_a = (max_a - min_a).prod()
    vol_b = (max_b - min_b).prod()
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0 else 0.0


def check_coverage(assignments, n_gaussians):
    assigned       = len(assignments)
    bg_count       = sum(1 for v in assignments.values() if v == 'background')
    object_assigned = assigned - bg_count
    coverage       = object_assigned / n_gaussians

    print(f'total gaussians:      {n_gaussians:,}')
    print(f'assigned (non-bg):    {object_assigned:,}  ({100*coverage:.1f}%)')
    print(f'background:           {bg_count:,}')
    print(f'unassigned:           {n_gaussians - assigned:,}')

    if coverage < MIN_COVERAGE:
        print(f'WARN  low coverage ({100*coverage:.1f}% < {100*MIN_COVERAGE:.0f}%)')
        return False
    print(f'PASS  coverage {100*coverage:.1f}%')
    return True


def check_compactness(objects, xyz):
    failures = []
    for obj_id, obj in objects.items():
        if obj['type'] == 'background':
            continue
        ids = obj['gaussian_ids']
        if len(ids) < 2:
            continue
        pts = xyz[ids]
        std = pts.std(axis=0).mean()
        if std > COMPACTNESS_WARN:
            failures.append((obj_id, obj['label'], std))

    if failures:
        print(f'WARN  {len(failures)} clusters have high position std (> {COMPACTNESS_WARN}m):')
        for obj_id, label, std in failures:
            print(f'      {obj_id} ({label}): std={std:.3f}m')
    else:
        print(f'PASS  all clusters spatially compact (std < {COMPACTNESS_WARN}m)')
    return len(failures) == 0


def check_bbox_separation(objects):
    obj_list = [(oid, obj) for oid, obj in objects.items()
                if obj['type'] == 'object']
    failures = []

    for i in range(len(obj_list)):
        for j in range(i + 1, len(obj_list)):
            id_a, obj_a = obj_list[i]
            id_b, obj_b = obj_list[j]
            iou = bbox_iou_3d(obj_a['bbox'], obj_b['bbox'])
            if iou > OVERLAP_WARN:
                failures.append((id_a, obj_a['label'], id_b, obj_b['label'], iou))

    if failures:
        print(f'WARN  {len(failures)} object bbox pairs have high overlap (IoU > {OVERLAP_WARN}):')
        for id_a, la, id_b, lb, iou in failures:
            print(f'      {id_a} ({la}) <-> {id_b} ({lb}): IoU={iou:.3f}')
    else:
        print(f'PASS  all object bboxes well separated')
    return len(failures) == 0


def visualize(objects, xyz):
    obj_ids  = [oid for oid in objects if objects[oid]['type'] != 'background']
    colors_map = plt.cm.tab20(np.linspace(0, 1, max(len(obj_ids), 1)))
    id_color = {oid: colors_map[i, :3] for i, oid in enumerate(obj_ids)}

    point_colors = np.full((len(xyz), 3), 0.5)  # gray for background

    for obj_id, obj in objects.items():
        if obj_id not in id_color:
            continue
        for idx in obj['gaussian_ids']:
            if idx < len(xyz):
                point_colors[idx] = id_color[obj_id]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # add bbox wireframes for each object
    geometries = [pcd]
    for obj_id, obj in objects.items():
        if obj['type'] == 'background':
            continue
        bbox = obj['bbox']
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox['min'], max_bound=bbox['max']
        )
        aabb.color = id_color.get(obj_id, [1, 0, 0])
        geometries.append(aabb)

    print('\nopen3d viewer — gray=background, colors=objects')
    print('objects:')
    for oid in obj_ids:
        obj = objects[oid]
        print(f'  {oid} ({obj["label"]}, {obj["type"]}): {len(obj["gaussian_ids"]):,} gaussians')

    o3d.visualization.draw_geometries(geometries, window_name='lifting validation')


def main():
    import matplotlib.pyplot as plt  # imported here so open3d doesn't conflict

    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussians',   required=True)
    parser.add_argument('--assignments', required=True)
    parser.add_argument('--objects',     required=True)
    args = parser.parse_args()

    xyz = load_gaussians(args.gaussians)
    N   = len(xyz)

    with open(args.assignments) as f:
        assignments = {int(k): v for k, v in json.load(f).items()}

    with open(args.objects) as f:
        objects = json.load(f)

    print(f'gaussians: {N:,}')
    print(f'objects:   {len(objects)}')
    print()

    passed = True
    passed &= check_coverage(assignments, N)
    print()
    passed &= check_compactness(objects, xyz)
    print()
    passed &= check_bbox_separation(objects)

    print()
    print('PASS' if passed else 'FAIL (see warnings above)')

    visualize(objects, xyz)


if __name__ == '__main__':
    main()
