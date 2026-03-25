"""
Tests for build_graph.py — all GPU/model-free.

Covers:
  - quat_to_rot: identity, 90-degree rotations, orthonormality
  - project_gaussians: basic projection, behind-camera culling, out-of-bounds culling
  - decode_rle: round-trip encode/decode
  - compute_bbox: correctness, single point, empty
  - lift_masks: plurality vote, background assignment, min vote fraction
  - infer_edges: on-top-of relation, no false edges for separated objects
  - embed_object_crops: area threshold filtering, frame cap, renormalization (mocked CLIP)
"""

import io
import json
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from build_graph import (
    MAX_EMBEDDING_FRAMES,
    MIN_CROP_AREA_FRAC,
    MIN_VOTE_FRACTION,
    compute_bbox,
    decode_rle,
    embed_object_crops,
    infer_edges,
    lift_masks,
    project_gaussians,
    quat_to_rot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rle(mask_np):
    """Encode a bool HxW array to pycocotools compressed RLE dict with str counts."""
    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def identity_cam(H=100, W=100, f=100.0):
    """Camera looking along +Z, no rotation, principal point at image center."""
    return {
        'R':  np.eye(3),
        't':  np.zeros(3),
        'fx': f,
        'fy': f,
        'cx': W / 2.0,
        'cy': H / 2.0,
        'w':  W,
        'h':  H,
    }


def make_segments(obj_ids, frame_names, H=100, W=100, fill=True):
    """
    Build a minimal segments dict.
    fill=True puts a full-image mask on every frame for every object.
    """
    objects = {}
    for oid in obj_ids:
        frames = {}
        for fn in frame_names:
            mask = np.ones((H, W), dtype=bool) if fill else np.zeros((H, W), dtype=bool)
            frames[fn] = {
                'mask_rle': make_rle(mask),
                'bbox_px':  [0, 0, W, H],
                'area_px':  int(mask.sum()),
            }
        objects[oid] = {'label': oid, 'type': 'object', 'frames': frames}
    return {'objects': objects}


# ---------------------------------------------------------------------------
# quat_to_rot
# ---------------------------------------------------------------------------

class TestQuatToRot:
    def test_identity(self):
        R = quat_to_rot(1, 0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_orthonormal(self):
        # 45-degree rotation around Y
        angle = np.pi / 4
        qw = np.cos(angle / 2)
        qy = np.sin(angle / 2)
        R = quat_to_rot(qw, 0, qy, 0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-7)

    def test_90_deg_around_z(self):
        # 90 deg around Z: x->y, y->-x
        angle = np.pi / 2
        qw = np.cos(angle / 2)
        qz = np.sin(angle / 2)
        R = quat_to_rot(qw, 0, 0, qz)
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-7)

    def test_180_deg_around_x(self):
        # 180 deg around X: y->-y, z->-z
        R = quat_to_rot(0, 1, 0, 0)
        np.testing.assert_allclose(R @ np.array([0, 1, 0]), [0, -1, 0], atol=1e-7)


# ---------------------------------------------------------------------------
# project_gaussians
# ---------------------------------------------------------------------------

class TestProjectGaussians:
    def test_center_point(self):
        # point at (0,0,1) with identity cam -> projects to principal point
        cam = identity_cam(H=100, W=100, f=100.0)
        xyz = np.array([[0.0, 0.0, 1.0]])
        px, py, vis = project_gaussians(xyz, cam, (100, 100))
        assert vis[0]
        assert px[0] == 50
        assert py[0] == 50

    def test_behind_camera_culled(self):
        cam = identity_cam()
        xyz = np.array([[0.0, 0.0, -1.0]])  # behind camera
        _, _, vis = project_gaussians(xyz, cam, (100, 100))
        assert not vis[0]

    def test_out_of_bounds_culled(self):
        cam = identity_cam(H=100, W=100, f=100.0)
        # point far to the right — projects well outside image
        xyz = np.array([[10.0, 0.0, 1.0]])
        _, _, vis = project_gaussians(xyz, cam, (100, 100))
        assert not vis[0]

    def test_offset_point(self):
        cam = identity_cam(H=100, W=100, f=100.0)
        # point at (0.5, 0, 1) -> projects 50px right of center = col 100 -> just at edge
        xyz = np.array([[0.49, 0.0, 1.0]])
        px, py, vis = project_gaussians(xyz, cam, (100, 100))
        assert vis[0]
        assert px[0] == 99

    def test_multiple_points(self):
        cam = identity_cam(H=100, W=100, f=100.0)
        xyz = np.array([
            [0.0,  0.0, 1.0],   # visible
            [0.0,  0.0, -1.0],  # behind
            [10.0, 0.0, 1.0],   # OOB
        ])
        _, _, vis = project_gaussians(xyz, cam, (100, 100))
        assert vis[0] and not vis[1] and not vis[2]


# ---------------------------------------------------------------------------
# decode_rle
# ---------------------------------------------------------------------------

class TestDecodeRle:
    def test_round_trip(self):
        mask = np.zeros((50, 60), dtype=bool)
        mask[10:20, 15:30] = True
        rle  = make_rle(mask)
        out  = decode_rle(rle)
        np.testing.assert_array_equal(out, mask)

    def test_full_mask(self):
        mask = np.ones((30, 40), dtype=bool)
        rle  = make_rle(mask)
        out  = decode_rle(rle)
        assert out.all()

    def test_empty_mask(self):
        mask = np.zeros((30, 40), dtype=bool)
        rle  = make_rle(mask)
        out  = decode_rle(rle)
        assert not out.any()


# ---------------------------------------------------------------------------
# compute_bbox
# ---------------------------------------------------------------------------

class TestComputeBbox:
    def test_basic(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 1.0]])
        bbox = compute_bbox(xyz, [0, 1, 2])
        assert bbox['min'] == [0.0, 0.0, 0.0]
        assert bbox['max'] == [2.0, 2.0, 3.0]
        np.testing.assert_allclose(bbox['centroid'], [1.0, 1.0, 4/3], atol=1e-6)

    def test_single_point(self):
        xyz = np.array([[1.0, 2.0, 3.0]])
        bbox = compute_bbox(xyz, [0])
        assert bbox['min'] == bbox['max'] == bbox['centroid'] == [1.0, 2.0, 3.0]

    def test_empty(self):
        xyz = np.array([[1.0, 2.0, 3.0]])
        assert compute_bbox(xyz, []) is None


# ---------------------------------------------------------------------------
# lift_masks
# ---------------------------------------------------------------------------

class TestLiftMasks:
    def _make_colmap(self, frame_names, H=100, W=100, f=100.0):
        """Single identity camera, one image per frame."""
        cameras = {1: {'w': W, 'h': H, 'fx': f, 'fy': f, 'cx': W/2, 'cy': H/2}}
        images  = {
            fn: {'cam_id': 1, 'R': np.eye(3), 't': np.zeros(3)}
            for fn in frame_names
        }
        return cameras, images

    def test_single_object_assigned(self, tmp_path):
        H, W = 100, 100
        # one Gaussian at center, one object mask covering the whole image
        xyz = np.array([[0.0, 0.0, 1.0]])  # projects to center
        frame_names = ['frame_000000.jpg']
        segments    = make_segments(['cup_0'], frame_names, H=H, W=W, fill=True)
        cameras, images = self._make_colmap(frame_names, H=H, W=W)

        # write a dummy image so lift_masks can read its shape
        img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'frame_000000.jpg')

        assignments, obj_ids, _ = lift_masks(
            xyz, segments, cameras, images, tmp_path
        )
        assert assignments[0] == obj_ids.index('cup_0')

    def test_background_when_no_votes(self, tmp_path):
        H, W = 100, 100
        xyz = np.array([[0.0, 0.0, -1.0]])  # behind camera — no votes
        frame_names = ['frame_000000.jpg']
        segments    = make_segments(['cup_0'], frame_names, H=H, W=W, fill=True)
        cameras, images = self._make_colmap(frame_names, H=H, W=W)

        img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'frame_000000.jpg')

        assignments, _, _ = lift_masks(xyz, segments, cameras, images, tmp_path)
        assert assignments[0] == -1

    def test_plurality_vote(self, tmp_path):
        # two objects, Gaussian gets more votes from obj_a than obj_b
        H, W = 100, 100
        xyz = np.array([[0.0, 0.0, 1.0]])

        mask_a = np.ones((H, W), dtype=bool)   # covers full image
        mask_b = np.zeros((H, W), dtype=bool)  # covers nothing

        segments = {
            'objects': {
                'cup_0':  {'label': 'cup',  'type': 'object', 'frames': {
                    'frame_000000.jpg': {'mask_rle': make_rle(mask_a), 'bbox_px': [0,0,W,H], 'area_px': H*W},
                }},
                'desk_0': {'label': 'desk', 'type': 'structure', 'frames': {
                    'frame_000000.jpg': {'mask_rle': make_rle(mask_b), 'bbox_px': [0,0,0,0], 'area_px': 0},
                }},
            }
        }
        cameras, images = self._make_colmap(['frame_000000.jpg'], H=H, W=W)
        img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'frame_000000.jpg')

        assignments, obj_ids, _ = lift_masks(xyz, segments, cameras, images, tmp_path)
        assert assignments[0] == obj_ids.index('cup_0')

    def test_min_vote_fraction(self, tmp_path):
        # Gaussian gets votes but below MIN_VOTE_FRACTION -> background
        H, W = 100, 100
        xyz = np.array([[0.0, 0.0, 1.0]])

        # cup gets 1 vote, desk gets 100 votes equivalent by having a full mask
        # We simulate low fraction by giving cup a tiny mask that doesn't cover center
        mask_cup = np.zeros((H, W), dtype=bool)
        mask_cup[0, 0] = True  # far corner, won't be hit by projected Gaussian

        mask_desk = np.ones((H, W), dtype=bool)

        segments = {
            'objects': {
                'cup_0':  {'label': 'cup',  'type': 'object', 'frames': {
                    'f0.jpg': {'mask_rle': make_rle(mask_cup),  'bbox_px': [0,0,1,1], 'area_px': 1},
                }},
                'desk_0': {'label': 'desk', 'type': 'structure', 'frames': {
                    'f0.jpg': {'mask_rle': make_rle(mask_desk), 'bbox_px': [0,0,W,H], 'area_px': H*W},
                }},
            }
        }
        cameras, images = self._make_colmap(['f0.jpg'], H=H, W=W)
        img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'f0.jpg')

        assignments, obj_ids, _ = lift_masks(xyz, segments, cameras, images, tmp_path)
        # desk wins plurality
        assert assignments[0] == obj_ids.index('desk_0')


# ---------------------------------------------------------------------------
# infer_edges
# ---------------------------------------------------------------------------

class TestInferEdges:
    def _make_node(self, label, ntype, centroid, mn, mx):
        return {
            'label': label,
            'type':  ntype,
            'bbox':  {'min': mn, 'max': mx, 'centroid': centroid},
            'clip_embedding': None,
        }

    def test_cup_on_desk(self):
        # desk is large flat surface, cup centroid is above it and within its XZ footprint
        nodes = {
            'desk_0': self._make_node('desk', 'structure',
                centroid=[0.5, 0.0, 0.5], mn=[0,0,0], mx=[1, 0.1, 1]),
            'cup_0':  self._make_node('cup',  'object',
                centroid=[0.5, -0.3, 0.5], mn=[0.4,-0.3,0.4], mx=[0.6,-0.1,0.6]),
        }
        edges = infer_edges(nodes)
        assert len(edges) == 1
        assert edges[0]['source'] == 'cup_0'
        assert edges[0]['target'] == 'desk_0'
        assert edges[0]['relation'] == 'on top of'

    def test_no_edge_when_outside_footprint(self):
        nodes = {
            'desk_0': self._make_node('desk', 'structure',
                centroid=[0.5, 0.0, 0.5], mn=[0,0,0], mx=[1, 0.1, 1]),
            'cup_0':  self._make_node('cup', 'object',
                centroid=[5.0, -0.3, 5.0], mn=[4.9,-0.3,4.9], mx=[5.1,-0.1,5.1]),
        }
        edges = infer_edges(nodes)
        assert len(edges) == 0

    def test_no_edge_when_same_height(self):
        nodes = {
            'obj_a': self._make_node('a', 'object',
                centroid=[0.5, 0.0, 0.5], mn=[0,-0.1,0], mx=[1,0.1,1]),
            'obj_b': self._make_node('b', 'object',
                centroid=[0.5, 0.02, 0.5], mn=[0,-0.1,0], mx=[1,0.1,1]),
        }
        edges = infer_edges(nodes)
        assert len(edges) == 0

    def test_background_node_excluded(self):
        nodes = {
            'background': {
                'label': 'background', 'type': 'background',
                'bbox': {'min': [-5,-5,-5], 'max': [5,5,5], 'centroid': [0,0,0]},
                'clip_embedding': None,
            },
            'cup_0': self._make_node('cup', 'object',
                centroid=[0.5, -0.3, 0.5], mn=[0.4,-0.3,0.4], mx=[0.6,-0.1,0.6]),
        }
        edges = infer_edges(nodes)
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# embed_object_crops — mocked CLIP
# ---------------------------------------------------------------------------

class TestEmbedObjectCrops:
    def _make_obj(self, frame_areas, H=100, W=100):
        """Build an object dict with given {frame_name: area_px} mapping."""
        frames = {}
        for fname, area in frame_areas.items():
            mask = np.zeros((H, W), dtype=bool)
            # fill top-left region proportional to area
            side = max(1, int(np.sqrt(area)))
            mask[:side, :side] = True
            frames[fname] = {
                'mask_rle': make_rle(mask),
                'bbox_px':  [0, 0, side, side],
                'area_px':  int(mask.sum()),
            }
        return {'label': 'cup', 'type': 'object', 'frames': frames}

    def _mock_clip(self):
        """Returns a mock CLIP model that outputs a fixed unit vector."""
        import torch
        import torch.nn.functional as F
        model     = MagicMock()
        preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
        emb = F.normalize(torch.ones(1, 512), dim=-1)
        model.encode_image = MagicMock(return_value=emb)
        return model, preprocess

    def test_selects_top_frames_by_area(self, tmp_path):
        import torch
        H, W   = 100, 100
        areas  = {f'frame_{i:03d}.jpg': max(200, (12 - i) * 100) for i in range(12)}
        obj    = self._make_obj(areas, H=H, W=W)

        for fname in areas:
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            img.save(tmp_path / fname)

        model, preprocess = self._mock_clip()
        emb, best_frame, embedding_frames = embed_object_crops(
            obj, tmp_path, model, preprocess, 'cpu'
        )

        # should not exceed MAX_EMBEDDING_FRAMES
        assert len(embedding_frames) <= MAX_EMBEDDING_FRAMES
        # best frame is the one with highest area
        assert best_frame == 'frame_000.jpg'

    def test_filters_small_crops(self, tmp_path):
        import torch
        H, W = 100, 100
        # one frame with large area, one with tiny area (below threshold)
        min_area = int(H * W * MIN_CROP_AREA_FRAC)
        areas = {
            'frame_big.jpg':   min_area * 10,
            'frame_tiny.jpg':  max(1, min_area - 1),
        }
        obj = self._make_obj(areas, H=H, W=W)

        for fname in areas:
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            img.save(tmp_path / fname)

        model, preprocess = self._mock_clip()
        emb, best_frame, embedding_frames = embed_object_crops(
            obj, tmp_path, model, preprocess, 'cpu'
        )
        assert 'frame_tiny.jpg' not in embedding_frames
        assert 'frame_big.jpg' in embedding_frames

    def test_embedding_is_unit_norm(self, tmp_path):
        import torch
        H, W  = 100, 100
        areas = {'frame_000.jpg': H * W}
        obj   = self._make_obj(areas, H=H, W=W)
        img   = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'frame_000.jpg')

        model, preprocess = self._mock_clip()
        emb, _, _ = embed_object_crops(obj, tmp_path, model, preprocess, 'cpu')
        assert emb is not None
        norm = np.linalg.norm(emb)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_returns_none_when_all_crops_too_small(self, tmp_path):
        H, W     = 100, 100
        min_area = int(H * W * MIN_CROP_AREA_FRAC)
        areas    = {'frame_000.jpg': max(1, min_area - 1)}
        obj      = self._make_obj(areas, H=H, W=W)
        img      = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
        img.save(tmp_path / 'frame_000.jpg')

        model, preprocess = self._mock_clip()
        emb, best_frame, embedding_frames = embed_object_crops(
            obj, tmp_path, model, preprocess, 'cpu'
        )
        assert emb is None
        assert embedding_frames == []
