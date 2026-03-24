"""
Validate scene graph CLIP embedding quality.

Checks:
  1. For each non-background node, cosine similarity between its label
     text and its CLIP embedding is above threshold
  2. For each node, correct label ranks highest against all other node labels
     (ranking test — diagonal of similarity matrix should be highest per row)

Prints full similarity matrix: rows=nodes, columns=all labels.

Usage:
    python pipeline/validate_graph.py \
        --graph pipeline/scene_graph.json

Input (scene_graph.json schema):
    {
      "nodes": {
        "<object_id>": {
          "label": str,
          "type": "object" | "structure" | "background",
          "clip_embedding": [512 floats] | null,
          "gaussian_ids": [int, ...],
          "bbox": {
            "min": [x, y, z],
            "max": [x, y, z],
            "centroid": [x, y, z]
          },
          "confidence": float | null
        }
      },
      "edges": [
        {
          "source": str,
          "target": str,
          "relation": str
        }
      ]
    }
"""

import argparse
import json

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

SIM_THRESHOLD = 0.2  # minimum cosine sim between node and its own label


def encode_texts(labels, device='cpu'):
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    with torch.no_grad():
        tokens = tokenizer(labels).to(device)
        embs   = model.encode_text(tokens)
        embs   = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def check_similarity_threshold(nodes, text_embs, label_list):
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    failures = []

    for node_id, node in nodes.items():
        if node['clip_embedding'] is None:
            continue
        emb  = np.array(node['clip_embedding'], dtype=np.float32)
        emb  = emb / (np.linalg.norm(emb) + 1e-6)
        label = node['label']
        if label not in label_to_idx:
            continue
        sim = float(emb @ text_embs[label_to_idx[label]])
        if sim < SIM_THRESHOLD:
            failures.append((node_id, label, sim))

    if failures:
        print(f'FAIL  {len(failures)} nodes below similarity threshold ({SIM_THRESHOLD}):')
        for node_id, label, sim in failures:
            print(f'      {node_id} ({label}): sim={sim:.3f}')
    else:
        print(f'PASS  all nodes above similarity threshold ({SIM_THRESHOLD})')
    return len(failures) == 0


def check_ranking(nodes, text_embs, label_list):
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    failures = []

    for node_id, node in nodes.items():
        if node['clip_embedding'] is None:
            continue
        emb   = np.array(node['clip_embedding'], dtype=np.float32)
        emb   = emb / (np.linalg.norm(emb) + 1e-6)
        label = node['label']
        if label not in label_to_idx:
            continue

        sims       = text_embs @ emb               # (n_labels,)
        ranked     = np.argsort(sims)[::-1]
        correct_rank = int(np.where(ranked == label_to_idx[label])[0][0])

        if correct_rank > 0:
            top_label = label_list[ranked[0]]
            failures.append((node_id, label, correct_rank, top_label, sims[label_to_idx[label]], sims[ranked[0]]))

    if failures:
        print(f'FAIL  {len(failures)} nodes where correct label is not top-ranked:')
        for node_id, label, rank, top, correct_sim, top_sim in failures:
            print(f'      {node_id} ({label}): rank={rank+1}, top="{top}"  '
                  f'correct={correct_sim:.3f} top={top_sim:.3f}')
    else:
        print(f'PASS  correct label is top-ranked for all nodes')
    return len(failures) == 0


def print_similarity_matrix(nodes, text_embs, label_list):
    node_ids  = [nid for nid, n in nodes.items() if n['clip_embedding'] is not None]
    node_embs = np.stack([
        np.array(nodes[nid]['clip_embedding'], dtype=np.float32) for nid in node_ids
    ])
    node_embs = node_embs / (np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-6)

    sim_matrix = node_embs @ text_embs.T  # (n_nodes, n_labels)

    col_w   = 10
    label_w = max(max(len(nid) + len(nodes[nid]['label']) + 3 for nid in node_ids), 20)

    header = f'{"node":<{label_w}}' + ''.join(f'{l[:col_w-1]:<{col_w}}' for l in label_list)
    print('\nSimilarity matrix (node embeddings vs label texts):')
    print(header)
    print('-' * len(header))

    for i, node_id in enumerate(node_ids):
        label = nodes[node_id]['label']
        row_label = f'{node_id} ({label})'
        row = f'{row_label:<{label_w}}'
        for j, sim in enumerate(sim_matrix[i]):
            marker = '*' if label_list[j] == label else ' '
            row += f'{sim:.3f}{marker:<{col_w-5}}'
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',  required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    with open(args.graph) as f:
        graph = json.load(f)

    nodes = graph['nodes']
    label_list = sorted(set(n['label'] for n in nodes.values() if n['clip_embedding'] is not None))

    print(f'nodes: {len(nodes)}  (with embeddings: {sum(1 for n in nodes.values() if n["clip_embedding"])})')
    print(f'labels: {label_list}')
    print()

    print('encoding label texts...')
    text_embs = encode_texts(label_list, device=args.device)
    print()

    passed = True
    passed &= check_similarity_threshold(nodes, text_embs, label_list)
    print()
    passed &= check_ranking(nodes, text_embs, label_list)

    print_similarity_matrix(nodes, text_embs, label_list)

    print()
    print('PASS' if passed else 'FAIL (see above)')


if __name__ == '__main__':
    main()
