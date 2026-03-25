"""
Tests for validate_lifting.py — no open3d, no PLY files needed.

Covers:
  - bbox_iou_3d: identical, no overlap, partial overlap, zero-volume bbox
  - check_coverage: above/below threshold, background counting, unassigned counting
  - check_compactness: compact cluster passes, spread cluster warns, single-point skip
  - check_bbox_separation: separated passes, overlapping warns, structure nodes excluded
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from validate_lifting import (
    bbox_iou_3d,
    check_bbox_separation,
    check_compactness,
    check_coverage,
)


# ---------------------------------------------------------------------------
# bbox_iou_3d
# ---------------------------------------------------------------------------

class TestBboxIou3d:
    def test_identical(self):
        bbox = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        assert bbox_iou_3d(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        b = {'min': [2, 2, 2], 'max': [3, 3, 3]}
        assert bbox_iou_3d(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {'min': [0, 0, 0], 'max': [2, 2, 2]}  # vol=8
        b = {'min': [1, 1, 1], 'max': [3, 3, 3]}  # vol=8, inter=1x1x1=1
        # union = 8 + 8 - 1 = 15
        assert bbox_iou_3d(a, b) == pytest.approx(1 / 15)

    def test_one_inside_other(self):
        outer = {'min': [0, 0, 0], 'max': [4, 4, 4]}  # vol=64
        inner = {'min': [1, 1, 1], 'max': [2, 2, 2]}  # vol=1
        # inter=1, union=64
        assert bbox_iou_3d(outer, inner) == pytest.approx(1 / 64)

    def test_zero_volume_bbox(self):
        a = {'min': [0, 0, 0], 'max': [0, 0, 0]}  # degenerate
        b = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        # union = 0 + 1 - 0 = 1, inter = 0
        assert bbox_iou_3d(a, b) == pytest.approx(0.0)

    def test_touching_but_not_overlapping(self):
        a = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        b = {'min': [1, 0, 0], 'max': [2, 1, 1]}
        # touching face, inter volume = 0
        assert bbox_iou_3d(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_coverage
# ---------------------------------------------------------------------------

class TestCheckCoverage:
    def test_above_threshold(self):
        # 60% non-background
        assignments = {i: 'cup' for i in range(60)}
        assignments.update({i: 'background' for i in range(60, 80)})
        assert check_coverage(assignments, n_gaussians=100) is True

    def test_below_threshold(self):
        # 40% non-background
        assignments = {i: 'cup' for i in range(40)}
        assignments.update({i: 'background' for i in range(40, 70)})
        assert check_coverage(assignments, n_gaussians=100) is False

    def test_all_background_fails(self):
        assignments = {i: 'background' for i in range(100)}
        assert check_coverage(assignments, n_gaussians=100) is False

    def test_all_object_passes(self):
        assignments = {i: 'cup' for i in range(100)}
        assert check_coverage(assignments, n_gaussians=100) is True

    def test_unassigned_counted_correctly(self):
        # 50 assigned as cup, 50 unassigned — coverage = 50/100 = 50% = threshold
        assignments = {i: 'cup' for i in range(50)}
        # exactly at MIN_COVERAGE=0.5 — should pass (>= not >)
        assert check_coverage(assignments, n_gaussians=100) is True


# ---------------------------------------------------------------------------
# check_compactness
# ---------------------------------------------------------------------------

class TestCheckCompactness:
    def _make_xyz(self, n=100):
        rng = np.random.default_rng(42)
        return rng.uniform(0, 10, (n, 3)).astype(np.float32)

    def test_compact_cluster_passes(self):
        xyz = np.zeros((100, 3), dtype=np.float32)
        # tight cluster: all within 0.1m
        xyz[:10] = np.random.default_rng(0).uniform(0, 0.1, (10, 3))
        objects = {
            'cup_0': {'type': 'object', 'label': 'cup', 'gaussian_ids': list(range(10))}
        }
        assert check_compactness(objects, xyz) is True

    def test_spread_cluster_warns(self):
        xyz = np.zeros((100, 3), dtype=np.float32)
        # spread cluster: points range over 10m
        xyz[:10, 0] = np.linspace(0, 10, 10)
        objects = {
            'cup_0': {'type': 'object', 'label': 'cup', 'gaussian_ids': list(range(10))}
        }
        assert check_compactness(objects, xyz) is False

    def test_background_skipped(self):
        xyz = np.zeros((100, 3), dtype=np.float32)
        xyz[:10, 0] = np.linspace(0, 10, 10)  # spread — would fail if checked
        objects = {
            'background': {'type': 'background', 'label': 'background', 'gaussian_ids': list(range(10))}
        }
        assert check_compactness(objects, xyz) is True

    def test_single_point_skipped(self):
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        objects = {
            'cup_0': {'type': 'object', 'label': 'cup', 'gaussian_ids': [0]}
        }
        assert check_compactness(objects, xyz) is True


# ---------------------------------------------------------------------------
# check_bbox_separation
# ---------------------------------------------------------------------------

class TestCheckBboxSeparation:
    def _make_obj(self, label, mn, mx, obj_type='object'):
        return {
            'type': obj_type,
            'label': label,
            'gaussian_ids': [],
            'bbox': {'min': mn, 'max': mx, 'centroid': [(a+b)/2 for a,b in zip(mn,mx)]}
        }

    def test_separated_objects_pass(self):
        objects = {
            'cup_0':  self._make_obj('cup',  [0,0,0], [0.1,0.1,0.1]),
            'desk_0': self._make_obj('desk', [1,0,1], [2,0.1,2]),
        }
        assert check_bbox_separation(objects) is True

    def test_overlapping_objects_warn(self):
        objects = {
            'cup_0':   self._make_obj('cup',   [0,0,0], [1,1,1]),
            'mug_0':   self._make_obj('mug',   [0,0,0], [1,1,1]),  # identical -> IoU=1.0
        }
        assert check_bbox_separation(objects) is False

    def test_structure_nodes_excluded(self):
        # structures with high overlap should not trigger warning
        objects = {
            'wall_0': self._make_obj('wall', [0,0,0], [3,3,3], obj_type='structure'),
            'floor_0': self._make_obj('floor', [0,0,0], [3,3,3], obj_type='structure'),
        }
        assert check_bbox_separation(objects) is True

    def test_mixed_structure_object(self):
        # cup overlapping with desk structure — structure excluded so should pass
        objects = {
            'cup_0':  self._make_obj('cup',  [0,0,0], [1,1,1], obj_type='object'),
            'desk_0': self._make_obj('desk', [0,0,0], [1,1,1], obj_type='structure'),
        }
        assert check_bbox_separation(objects) is True

    def test_single_object(self):
        objects = {
            'cup_0': self._make_obj('cup', [0,0,0], [1,1,1]),
        }
        assert check_bbox_separation(objects) is True
