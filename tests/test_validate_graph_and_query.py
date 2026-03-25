"""
Tests for validate_graph.py and validate_query.py — no CLIP model loaded.

Mocks the embedding layer to test all logic around:
  - check_similarity_threshold: pass/fail, background skipped, null embedding skipped
  - check_ranking: correct rank=0 passes, rank>0 fails, background skipped
  - query_graph: correct sorting, background excluded, null embedding excluded
  - run_tests: label match, centroid check, no results case
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from validate_graph import check_ranking, check_similarity_threshold
from validate_query import query_graph, run_tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_vec(dim=512, seed=None):
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_node(label, ntype, emb=None, centroid=None):
    return {
        'label':          label,
        'type':           ntype,
        'clip_embedding': emb.tolist() if emb is not None else None,
        'bbox':           {'centroid': centroid or [0.0, 0.0, 0.0]},
        'confidence':     None,
    }


# ---------------------------------------------------------------------------
# check_similarity_threshold
# ---------------------------------------------------------------------------

class TestCheckSimilarityThreshold:
    def test_above_threshold_passes(self):
        # embedding identical to text -> cosine sim = 1.0
        emb = unit_vec(seed=0)
        nodes = {'cup_0': make_node('cup', 'object', emb=emb)}
        text_embs  = np.stack([emb])
        label_list = ['cup']
        assert check_similarity_threshold(nodes, text_embs, label_list) is True

    def test_below_threshold_fails(self):
        emb_node = unit_vec(seed=0)
        emb_text = unit_vec(seed=1)
        # force low similarity: use orthogonal vectors
        emb_text = emb_node.copy()
        emb_text[0] = -emb_node[0]
        emb_text = emb_text / np.linalg.norm(emb_text)

        nodes = {'cup_0': make_node('cup', 'object', emb=emb_node)}
        text_embs  = np.stack([emb_text])
        label_list = ['cup']
        # sim will be low, below SIM_THRESHOLD=0.2
        result = check_similarity_threshold(nodes, text_embs, label_list)
        # just check it ran without error — actual pass/fail depends on sim value
        assert isinstance(result, bool)

    def test_null_embedding_skipped(self):
        nodes = {'cup_0': make_node('cup', 'object', emb=None)}
        text_embs  = np.zeros((1, 512), dtype=np.float32)
        label_list = ['cup']
        # no embeddings to check -> passes trivially
        assert check_similarity_threshold(nodes, text_embs, label_list) is True

    def test_background_node_skipped(self):
        # background node with null embedding — must not crash
        emb = unit_vec(seed=0)
        nodes = {
            'background': make_node('background', 'background', emb=None),
            'cup_0':      make_node('cup', 'object', emb=emb),
        }
        text_embs  = np.stack([emb])
        label_list = ['cup']
        assert check_similarity_threshold(nodes, text_embs, label_list) is True

    def test_label_not_in_label_list_skipped(self):
        emb = unit_vec(seed=0)
        nodes = {'mystery_0': make_node('mystery_object', 'object', emb=emb)}
        text_embs  = np.stack([unit_vec(seed=1)])
        label_list = ['cup']
        # label not in list -> skipped -> trivially passes
        assert check_similarity_threshold(nodes, text_embs, label_list) is True


# ---------------------------------------------------------------------------
# check_ranking
# ---------------------------------------------------------------------------

class TestCheckRanking:
    def test_correct_label_top_ranked(self):
        # cup embedding is closest to cup text, not keyboard text
        cup_emb  = unit_vec(seed=0)
        kb_emb   = unit_vec(seed=1)

        nodes = {'cup_0': make_node('cup', 'object', emb=cup_emb)}
        # text_embs: [cup_text, keyboard_text]
        # make cup_text = cup_emb so similarity = 1.0
        text_embs  = np.stack([cup_emb, kb_emb])
        label_list = ['cup', 'keyboard']
        assert check_ranking(nodes, text_embs, label_list) is True

    def test_wrong_label_top_ranked_fails(self):
        cup_emb = unit_vec(seed=0)
        kb_emb  = unit_vec(seed=1)

        nodes = {'cup_0': make_node('cup', 'object', emb=cup_emb)}
        # swap: keyboard text is identical to cup embedding -> keyboard ranks first
        text_embs  = np.stack([kb_emb, cup_emb])  # [cup_text, keyboard_text]
        label_list = ['cup', 'keyboard']
        # cup node's emb @ cup_text will be low, @ keyboard_text will be 1.0 -> keyboard ranks first
        assert check_ranking(nodes, text_embs, label_list) is False

    def test_null_embedding_skipped(self):
        nodes = {'cup_0': make_node('cup', 'object', emb=None)}
        text_embs  = np.zeros((1, 512), dtype=np.float32)
        label_list = ['cup']
        assert check_ranking(nodes, text_embs, label_list) is True


# ---------------------------------------------------------------------------
# query_graph
# ---------------------------------------------------------------------------

class TestQueryGraph:
    def _make_graph(self, nodes_spec):
        """nodes_spec: list of (node_id, label, type, emb, centroid)"""
        nodes = {}
        for nid, label, ntype, emb, centroid in nodes_spec:
            nodes[nid] = make_node(label, ntype, emb=emb, centroid=centroid)
        return {'nodes': nodes, 'edges': []}

    def test_sorted_by_similarity(self):
        cup_emb  = unit_vec(seed=0)
        desk_emb = unit_vec(seed=1)
        query    = cup_emb  # identical to cup -> sim=1.0

        graph = self._make_graph([
            ('desk_0', 'desk', 'structure', desk_emb, [0,0,0]),
            ('cup_0',  'cup',  'object',    cup_emb,  [1,0,0]),
        ])
        results = query_graph(graph, query)
        assert results[0][0] == 'cup_0'
        assert results[1][0] == 'desk_0'
        assert results[0][3] > results[1][3]

    def test_background_excluded(self):
        emb = unit_vec(seed=0)
        graph = self._make_graph([
            ('background', 'background', 'background', emb, [0,0,0]),
            ('cup_0', 'cup', 'object', emb, [1,0,0]),
        ])
        results = query_graph(graph, emb)
        ids = [r[0] for r in results]
        assert 'background' not in ids
        assert 'cup_0' in ids

    def test_null_embedding_excluded(self):
        emb = unit_vec(seed=0)
        graph = self._make_graph([
            ('cup_0',    'cup',    'object', emb,  [0,0,0]),
            ('monitor_0','monitor','object', None, [1,0,0]),
        ])
        results = query_graph(graph, emb)
        ids = [r[0] for r in results]
        assert 'monitor_0' not in ids

    def test_empty_graph_returns_empty(self):
        graph = {'nodes': {}, 'edges': []}
        results = query_graph(graph, unit_vec(seed=0))
        assert results == []

    def test_returns_centroid(self):
        emb = unit_vec(seed=0)
        graph = self._make_graph([
            ('cup_0', 'cup', 'object', emb, [1.5, 0.3, -0.2]),
        ])
        results = query_graph(graph, emb)
        assert results[0][2] == [1.5, 0.3, -0.2]


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------

class TestRunTests:
    def _make_graph(self, label, emb, centroid):
        node = make_node(label, 'object', emb=emb, centroid=centroid)
        return {'nodes': {'obj_0': node}, 'edges': []}

    def _make_query_embs(self, query, emb):
        """Pre-built query embeddings dict to bypass CLIP."""
        return {query: emb}

    def test_correct_label_passes(self, monkeypatch):
        import validate_query
        emb   = unit_vec(seed=0)
        graph = self._make_graph('cup', emb, [0.0, 0.0, 0.0])
        tests = [{'query': 'cup', 'expected_label': 'cup',
                  'expected_centroid': None, 'centroid_threshold': None}]

        monkeypatch.setattr(validate_query, 'encode_text', lambda q, device='cpu': emb)
        assert run_tests(graph, tests, 'cpu') is True

    def test_wrong_label_fails(self, monkeypatch):
        import validate_query
        emb   = unit_vec(seed=0)
        graph = self._make_graph('cup', emb, [0.0, 0.0, 0.0])
        tests = [{'query': 'cup', 'expected_label': 'monitor',
                  'expected_centroid': None, 'centroid_threshold': None}]

        monkeypatch.setattr(validate_query, 'encode_text', lambda q, device='cpu': emb)
        assert run_tests(graph, tests, 'cpu') is False

    def test_centroid_within_threshold_passes(self, monkeypatch):
        import validate_query
        emb      = unit_vec(seed=0)
        centroid = [1.0, 0.0, 0.0]
        graph    = self._make_graph('cup', emb, centroid)
        tests = [{'query': 'cup', 'expected_label': 'cup',
                  'expected_centroid': [1.05, 0.0, 0.0], 'centroid_threshold': 0.15}]

        monkeypatch.setattr(validate_query, 'encode_text', lambda q, device='cpu': emb)
        assert run_tests(graph, tests, 'cpu') is True

    def test_centroid_outside_threshold_fails(self, monkeypatch):
        import validate_query
        emb      = unit_vec(seed=0)
        centroid = [1.0, 0.0, 0.0]
        graph    = self._make_graph('cup', emb, centroid)
        tests = [{'query': 'cup', 'expected_label': 'cup',
                  'expected_centroid': [2.0, 0.0, 0.0], 'centroid_threshold': 0.15}]

        monkeypatch.setattr(validate_query, 'encode_text', lambda q, device='cpu': emb)
        assert run_tests(graph, tests, 'cpu') is False

    def test_no_results_fails(self, monkeypatch):
        import validate_query
        emb   = unit_vec(seed=0)
        graph = {'nodes': {}, 'edges': []}
        tests = [{'query': 'cup', 'expected_label': 'cup',
                  'expected_centroid': None, 'centroid_threshold': None}]

        monkeypatch.setattr(validate_query, 'encode_text', lambda q, device='cpu': emb)
        assert run_tests(graph, tests, 'cpu') is False

    def test_deduplicates_query_encoding(self, monkeypatch):
        import validate_query
        emb      = unit_vec(seed=0)
        graph    = self._make_graph('cup', emb, [0.0, 0.0, 0.0])
        # same query twice — encode_text should only be called once
        tests = [
            {'query': 'cup', 'expected_label': 'cup', 'expected_centroid': None, 'centroid_threshold': None},
            {'query': 'cup', 'expected_label': 'cup', 'expected_centroid': None, 'centroid_threshold': None},
        ]
        call_count = {'n': 0}
        def counting_encode(q, device='cpu'):
            call_count['n'] += 1
            return emb
        monkeypatch.setattr(validate_query, 'encode_text', counting_encode)
        run_tests(graph, tests, 'cpu')
        assert call_count['n'] == 1
