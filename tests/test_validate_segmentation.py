"""
Tests for validate_segmentation.py — no GPU, no display.

Covers:
  - rle_decode: compressed string RLE, uncompressed list RLE
  - compute_iou: identical masks, no overlap, partial overlap, empty masks
  - check_missed_nouns: pass/fail cases
  - check_id_consistency: consistent IDs pass, inconsistent fail, single-frame skip
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from validate_segmentation import check_id_consistency, check_missed_nouns, compute_iou, rle_decode


def make_compressed_rle(mask_np):
    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return {'counts': rle['counts'], 'size': list(mask_np.shape)}


# ---------------------------------------------------------------------------
# rle_decode
# ---------------------------------------------------------------------------

class TestRleDecode:
    def test_compressed_string_round_trip(self):
        mask = np.zeros((50, 60), dtype=bool)
        mask[5:15, 10:30] = True
        rle = make_compressed_rle(mask)
        out = rle_decode(rle, mask.shape)
        np.testing.assert_array_equal(out, mask)

    def test_uncompressed_list(self):
        # uncompressed RLE: [0-count, 1-count, 0-count, ...]
        # 3 zeros then 2 ones then 5 zeros = 10 pixels
        rle = {'counts': [3, 2, 5], 'size': [2, 5]}
        out = rle_decode(rle, (2, 5))
        flat = out.flatten()
        assert not flat[0] and not flat[1] and not flat[2]
        assert flat[3] and flat[4]
        assert not any(flat[5:])

    def test_full_mask(self):
        mask = np.ones((20, 30), dtype=bool)
        rle  = make_compressed_rle(mask)
        out  = rle_decode(rle, mask.shape)
        assert out.all()

    def test_empty_mask(self):
        mask = np.zeros((20, 30), dtype=bool)
        rle  = make_compressed_rle(mask)
        out  = rle_decode(rle, mask.shape)
        assert not out.any()


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

class TestComputeIou:
    def test_identical_masks(self):
        mask = np.ones((10, 10), dtype=bool)
        assert compute_iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5, :] = True
        b[5:, :] = True
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:, :5] = True   # left half
        b[:, 5:] = True   # right half — no overlap, but let's do half overlap
        # reset: a covers cols 0-7, b covers cols 4-9
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:, :8] = True
        b[:, 4:] = True
        # intersection: cols 4-7 = 4 cols, union: cols 0-9 = 10 cols
        expected = (4 * 10) / (10 * 10)
        assert compute_iou(a, b) == pytest.approx(expected)

    def test_both_empty(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_one_empty(self):
        a = np.ones((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        assert compute_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_missed_nouns
# ---------------------------------------------------------------------------

class TestCheckMissedNouns:
    def test_no_missed(self):
        segments = {'missed_nouns': []}
        assert check_missed_nouns(segments) is True

    def test_with_missed(self):
        segments = {'missed_nouns': ['shelf', 'plant']}
        assert check_missed_nouns(segments) is False

    def test_missing_key_defaults_to_pass(self):
        segments = {}
        assert check_missed_nouns(segments) is True


# ---------------------------------------------------------------------------
# check_id_consistency
# ---------------------------------------------------------------------------

class TestCheckIdConsistency:
    def _make_segments(self, obj_id, frame_masks):
        """frame_masks: list of (frame_name, np.bool_ mask)"""
        frames = {}
        for fname, mask in frame_masks:
            rle = make_compressed_rle(mask)
            frames[fname] = {'mask_rle': rle, 'bbox_px': [0,0,1,1], 'area_px': 1}
        return {
            'objects': {
                obj_id: {'label': 'cup', 'type': 'object', 'frames': frames}
            }
        }

    def test_identical_masks_pass(self):
        mask = np.ones((20, 20), dtype=bool)
        segments = self._make_segments('cup_0', [
            ('f0.jpg', mask), ('f1.jpg', mask), ('f2.jpg', mask)
        ])
        assert check_id_consistency(segments, iou_threshold=0.5) is True

    def test_no_overlap_fails(self):
        mask_a = np.zeros((20, 20), dtype=bool)
        mask_b = np.zeros((20, 20), dtype=bool)
        mask_a[:10, :] = True
        mask_b[10:, :] = True
        segments = self._make_segments('cup_0', [('f0.jpg', mask_a), ('f1.jpg', mask_b)])
        assert check_id_consistency(segments, iou_threshold=0.5) is False

    def test_single_frame_skipped(self):
        mask = np.ones((20, 20), dtype=bool)
        segments = self._make_segments('cup_0', [('f0.jpg', mask)])
        # only one frame — no adjacent pair to check, should pass
        assert check_id_consistency(segments, iou_threshold=0.5) is True

    def test_partial_overlap_above_threshold_passes(self):
        # overlap ~80% should pass at threshold 0.5
        mask_a = np.zeros((10, 10), dtype=bool)
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_a[:, :8] = True   # 80 px
        mask_b[:, 2:] = True   # 80 px, overlap cols 2-7 = 60 px
        # IoU = 60 / (80 + 80 - 60) = 60/100 = 0.6
        segments = self._make_segments('cup_0', [('f0.jpg', mask_a), ('f1.jpg', mask_b)])
        assert check_id_consistency(segments, iou_threshold=0.5) is True

    def test_multiple_objects_independent(self):
        good_mask = np.ones((20, 20), dtype=bool)
        bad_a = np.zeros((20, 20), dtype=bool)
        bad_b = np.zeros((20, 20), dtype=bool)
        bad_a[:10, :] = True
        bad_b[10:, :] = True

        segments = {
            'objects': {
                'cup_0': {
                    'label': 'cup', 'type': 'object',
                    'frames': {
                        'f0.jpg': {'mask_rle': make_compressed_rle(good_mask), 'bbox_px': [0,0,1,1], 'area_px': 1},
                        'f1.jpg': {'mask_rle': make_compressed_rle(good_mask), 'bbox_px': [0,0,1,1], 'area_px': 1},
                    }
                },
                'desk_0': {
                    'label': 'desk', 'type': 'structure',
                    'frames': {
                        'f0.jpg': {'mask_rle': make_compressed_rle(bad_a), 'bbox_px': [0,0,1,1], 'area_px': 1},
                        'f1.jpg': {'mask_rle': make_compressed_rle(bad_b), 'bbox_px': [0,0,1,1], 'area_px': 1},
                    }
                },
            }
        }
        # desk_0 fails, so overall should fail
        assert check_id_consistency(segments, iou_threshold=0.5) is False
