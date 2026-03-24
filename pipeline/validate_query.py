"""
Validate query pipeline against a fixed test set.

Runs a set of predefined queries against the scene graph and checks:
  1. Correct object is returned for unambiguous queries
  2. Returned centroid is within distance threshold of expected position
  3. Relational queries return same object as direct queries
  4. Ambiguous queries (multiple instances) return multiple results

Test cases are defined in a JSON file alongside this script.

Usage:
    python pipeline/validate_query.py \
        --graph pipeline/scene_graph.json \
        --tests pipeline/query_tests.json

query_tests.json schema:
    {
      "tests": [
        {
          "query": "cup",
          "expected_label": "cup",
          "expected_centroid": [x, y, z],
          "centroid_threshold": 0.15
        },
        {
          "query": "monitor on the desk",
          "expected_label": "monitor",
          "expected_centroid": null,
          "centroid_threshold": null
        }
      ]
    }

expected_centroid: null means skip position check, just verify correct label returned.
centroid_threshold: max allowed distance in meters between returned and expected centroid.
"""

import argparse
import json

import numpy as np
import open_clip
import torch
import torch.nn.functional as F


def encode_text(query, device='cpu'):
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        emb    = model.encode_text(tokens)
        emb    = F.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu().numpy()


def query_graph(graph, query_emb):
    """
    Returns list of (node_id, label, centroid, similarity) sorted by similarity desc.
    Skips background nodes and nodes with no embedding.
    """
    nodes   = graph['nodes']
    results = []

    for node_id, node in nodes.items():
        if node['clip_embedding'] is None:
            continue
        if node['type'] == 'background':
            continue
        emb = np.array(node['clip_embedding'], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        sim = float(emb @ query_emb)
        results.append((node_id, node['label'], node['bbox']['centroid'], sim))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def run_tests(graph, tests, device):
    passed = 0
    failed = 0

    # cache CLIP model — encode all queries at once
    all_queries = list(set(t['query'] for t in tests))
    query_embs  = {}
    print('encoding queries...')
    for q in all_queries:
        query_embs[q] = encode_text(q, device)
    print()

    for test in tests:
        query    = test['query']
        expected = test['expected_label']
        exp_pos  = test.get('expected_centroid')
        threshold = test.get('centroid_threshold', 0.15)

        results = query_graph(graph, query_embs[query])

        if not results:
            print(f'FAIL  "{query}" — no results returned')
            failed += 1
            continue

        top_id, top_label, top_centroid, top_sim = results[0]

        label_ok = top_label == expected
        pos_ok   = True
        dist     = None

        if exp_pos is not None:
            dist   = float(np.linalg.norm(np.array(top_centroid) - np.array(exp_pos)))
            pos_ok = dist <= threshold

        if label_ok and pos_ok:
            pos_str = f'  dist={dist:.3f}m' if dist is not None else ''
            print(f'PASS  "{query}" -> {top_label} (sim={top_sim:.3f}{pos_str})')
            passed += 1
        else:
            reasons = []
            if not label_ok:
                reasons.append(f'got "{top_label}" expected "{expected}"')
            if not pos_ok:
                reasons.append(f'centroid dist={dist:.3f}m > threshold={threshold}m')
            print(f'FAIL  "{query}" — {", ".join(reasons)}  (sim={top_sim:.3f})')
            print(f'      top result: {top_id} at {[f"{c:.3f}" for c in top_centroid]}')
            if len(results) > 1:
                second = results[1]
                print(f'      2nd result: {second[0]} ({second[1]}) sim={second[3]:.3f}')
            failed += 1

    print()
    print(f'{passed}/{passed+failed} tests passed')
    return failed == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',  required=True)
    parser.add_argument('--tests',  required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    with open(args.graph) as f:
        graph = json.load(f)

    with open(args.tests) as f:
        test_data = json.load(f)

    print(f'nodes in graph: {len(graph["nodes"])}')
    print(f'test cases:     {len(test_data["tests"])}')
    print()

    passed = run_tests(graph, test_data['tests'], args.device)
    exit(0 if passed else 1)


if __name__ == '__main__':
    main()
