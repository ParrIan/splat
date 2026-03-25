"""
Tests for segment_masks.py — all GPU/model-free.

Covers:
  - sorted_frame_names: ordering, extension filtering, empty dir error
  - mask_to_rle: round-trip, all-true, all-false
  - bbox_from_mask: basic, empty mask, single pixel, non-square
  - build_segments: noun-to-type mapping, missed nouns, object ID format, empty frame data dropped
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from segment_masks import bbox_from_mask, build_segments, mask_to_rle, sorted_frame_names


# ---------------------------------------------------------------------------
# sorted_frame_names
# ---------------------------------------------------------------------------

class TestSortedFrameNames:
    def test_sorted_order(self, tmp_path):
        for name in ['frame_000010.jpg', 'frame_000002.jpg', 'frame_000001.jpg']:
            (tmp_path / name).touch()
        names = sorted_frame_names(tmp_path)
        assert names == ['frame_000001.jpg', 'frame_000002.jpg', 'frame_000010.jpg']

    def test_filters_non_image_files(self, tmp_path):
        (tmp_path / 'frame_000001.jpg').touch()
        (tmp_path / 'readme.txt').touch()
        (tmp_path / 'data.json').touch()
        names = sorted_frame_names(tmp_path)
        assert names == ['frame_000001.jpg']

    def test_accepts_png_and_jpeg(self, tmp_path):
        (tmp_path / 'a.jpg').touch()
        (tmp_path / 'b.jpeg').touch()
        (tmp_path / 'c.png').touch()
        names = sorted_frame_names(tmp_path)
        assert set(names) == {'a.jpg', 'b.jpeg', 'c.png'}

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match='no images found'):
            sorted_frame_names(tmp_path)

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / 'frame.JPG').touch()
        (tmp_path / 'frame2.PNG').touch()
        names = sorted_frame_names(tmp_path)
        assert len(names) == 2


# ---------------------------------------------------------------------------
# mask_to_rle
# ---------------------------------------------------------------------------

class TestMaskToRle:
    def test_round_trip(self):
        mask = np.zeros((60, 80), dtype=bool)
        mask[10:30, 20:50] = True
        rle  = mask_to_rle(mask)
        decoded = mask_utils.decode(
            {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
        ).astype(bool)
        np.testing.assert_array_equal(decoded, mask)

    def test_counts_is_string(self):
        mask = np.ones((10, 10), dtype=bool)
        rle  = mask_to_rle(mask)
        assert isinstance(rle['counts'], str)

    def test_size_matches_mask(self):
        mask = np.zeros((40, 60), dtype=bool)
        rle  = mask_to_rle(mask)
        assert rle['size'] == [40, 60]

    def test_empty_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        rle  = mask_to_rle(mask)
        decoded = mask_utils.decode(
            {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
        ).astype(bool)
        assert not decoded.any()


# ---------------------------------------------------------------------------
# bbox_from_mask
# ---------------------------------------------------------------------------

class TestBboxFromMask:
    def test_basic(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 30:50] = True
        x0, y0, x1, y1 = bbox_from_mask(mask)
        assert x0 == 30 and y0 == 10
        assert x1 == 49 and y1 == 19

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=bool)
        assert bbox_from_mask(mask) == [0, 0, 0, 0]

    def test_single_pixel(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 30] = True
        x0, y0, x1, y1 = bbox_from_mask(mask)
        assert x0 == x1 == 30
        assert y0 == y1 == 25

    def test_full_mask(self):
        mask = np.ones((40, 60), dtype=bool)
        x0, y0, x1, y1 = bbox_from_mask(mask)
        assert x0 == 0 and y0 == 0
        assert x1 == 59 and y1 == 39

    def test_non_square(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[40:60, 80:120] = True
        x0, y0, x1, y1 = bbox_from_mask(mask)
        assert x0 == 80 and x1 == 119
        assert y0 == 40 and y1 == 59


# ---------------------------------------------------------------------------
# build_segments
# ---------------------------------------------------------------------------

class TestBuildSegments:
    def _make_mock_models(self, detections_per_noun):
        """
        detections_per_noun: dict {noun: n_instances} or {noun: 0} for missed
        Returns mock sam2, mock gdino that produce synthetic results.
        """
        H, W = 50, 50

        def fake_run_noun(sam2, gdino, image_dir, frame_names, noun, device):
            n = detections_per_noun.get(noun, 0)
            if n == 0:
                return {}
            results = {}
            for inst_idx in range(n):
                mask = np.zeros((H, W), dtype=bool)
                mask[10:20, 10:20] = True
                rle  = mask_to_rle(mask)
                results[inst_idx] = {
                    fn: {'mask_rle': rle, 'bbox_px': [10,10,20,20], 'area_px': 100}
                    for fn in frame_names
                }
            return results

        return MagicMock(), MagicMock(), fake_run_noun

    def test_objects_and_structures_typed_correctly(self, tmp_path, monkeypatch):
        import segment_masks
        nouns = {'objects': ['cup'], 'structures': ['desk']}
        frame_names = ['f0.jpg', 'f1.jpg']

        sam2, gdino, fake_run = self._make_mock_models({'cup': 1, 'desk': 1})
        monkeypatch.setattr(segment_masks, 'run_noun_sam2', fake_run)

        objects, missed = segment_masks.build_segments(
            sam2, gdino, tmp_path, frame_names, nouns, 'cpu'
        )
        assert objects['cup_0']['type'] == 'object'
        assert objects['desk_0']['type'] == 'structure'
        assert missed == []

    def test_missed_noun_recorded(self, tmp_path, monkeypatch):
        import segment_masks
        nouns = {'objects': ['cup', 'monitor'], 'structures': []}
        frame_names = ['f0.jpg']

        sam2, gdino, fake_run = self._make_mock_models({'cup': 1, 'monitor': 0})
        monkeypatch.setattr(segment_masks, 'run_noun_sam2', fake_run)

        objects, missed = segment_masks.build_segments(
            sam2, gdino, tmp_path, frame_names, nouns, 'cpu'
        )
        assert 'monitor' in missed
        assert 'cup_0' in objects

    def test_object_id_format(self, tmp_path, monkeypatch):
        import segment_masks
        nouns = {'objects': ['red cup'], 'structures': []}
        frame_names = ['f0.jpg']

        sam2, gdino, fake_run = self._make_mock_models({'red cup': 2})
        monkeypatch.setattr(segment_masks, 'run_noun_sam2', fake_run)

        objects, _ = segment_masks.build_segments(
            sam2, gdino, tmp_path, frame_names, nouns, 'cpu'
        )
        # spaces replaced with underscores
        assert 'red_cup_0' in objects
        assert 'red_cup_1' in objects

    def test_empty_frame_data_dropped(self, tmp_path, monkeypatch):
        import segment_masks
        nouns = {'objects': ['cup'], 'structures': []}
        frame_names = ['f0.jpg']

        def fake_run(sam2, gdino, image_dir, frame_names, noun, device):
            # instance with empty frame dict
            return {0: {}}

        sam2, gdino = MagicMock(), MagicMock()
        monkeypatch.setattr(segment_masks, 'run_noun_sam2', fake_run)

        objects, missed = segment_masks.build_segments(
            sam2, gdino, tmp_path, frame_names, nouns, 'cpu'
        )
        assert 'cup_0' not in objects
        assert 'cup' in missed

    def test_multiple_instances(self, tmp_path, monkeypatch):
        import segment_masks
        nouns = {'objects': ['cup'], 'structures': []}
        frame_names = ['f0.jpg']

        sam2, gdino, fake_run = self._make_mock_models({'cup': 3})
        monkeypatch.setattr(segment_masks, 'run_noun_sam2', fake_run)

        objects, _ = segment_masks.build_segments(
            sam2, gdino, tmp_path, frame_names, nouns, 'cpu'
        )
        assert len(objects) == 3
        assert all(f'cup_{i}' in objects for i in range(3))
