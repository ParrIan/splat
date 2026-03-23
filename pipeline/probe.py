"""
Feature field probe — visualize CLIP similarity for text queries.

Usage:
    python probe.py --ply stool_colmap_features.ply --mlp stool_colmap_feature_mlp.pt --query "stool"
    python probe.py --ply stool_colmap_features.ply --mlp stool_colmap_feature_mlp.pt --query "floor" --top 0.05
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import open_clip
import open3d as o3d
from plyfile import PlyData

CLIP_DIM = 512
FEATURE_DIM = 64


class FeatureDistillationMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def load_features_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)

    # prefer raw 512-dim features for text query matching
    raw_cols = sorted([p.name for p in v.properties if p.name.startswith("f_raw_")],
                      key=lambda s: int(s.split("_")[-1]))
    clip_cols = sorted([p.name for p in v.properties if p.name.startswith("f_clip_")],
                       key=lambda s: int(s.split("_")[-1]))

    if raw_cols:
        features = np.stack([v[c] for c in raw_cols], axis=1).astype(np.float32)
        print(f"loaded {len(xyz):,} gaussians, {len(raw_cols)}-dim raw features")
        return xyz, features, "raw"
    elif clip_cols:
        features = np.stack([v[c] for c in clip_cols], axis=1).astype(np.float32)
        print(f"loaded {len(xyz):,} gaussians, {len(clip_cols)}-dim compressed features (no raw)")
        return xyz, features, "compressed"
    else:
        raise ValueError("no f_raw_* or f_clip_* properties found — wrong PLY file?")


def encode_text(query, device="cpu"):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        text_feat = model.encode_text(tokens)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    return text_feat.squeeze(0)  # (512,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--mlp", default=None, help="optional legacy MLP weights (not needed for DINOv2 pipeline)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top", type=float, default=0.05, help="top fraction to highlight")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    xyz, features, feat_type = load_features_ply(args.ply)
    features_t = torch.from_numpy(features)
    # normalize rows
    features_t = torch.nn.functional.normalize(features_t, dim=-1)

    zero_mask = (features_t.norm(dim=1) < 1e-6)
    print(f"gaussians with no feature: {zero_mask.sum().item():,} / {len(xyz):,}")

    print(f"encoding query: '{args.query}'")
    text_feat = encode_text(args.query, device=args.device)  # (512,)

    with torch.no_grad():
        # f_clip_* at 512-dim = CLIP-aligned (DINOv2 pipeline) — compare directly
        # f_raw_* at 512-dim = raw aggregated CLIP patches — compare directly
        # f_clip_* at 64-dim = compressed (legacy) — needs MLP
        if features_t.shape[1] == CLIP_DIM:
            sims = features_t @ text_feat
        else:
            assert args.mlp is not None, "64-dim compressed features require --mlp"
            mlp = FeatureDistillationMLP(in_dim=CLIP_DIM, out_dim=FEATURE_DIM)
            mlp.load_state_dict(torch.load(args.mlp, map_location="cpu"))
            mlp.eval()
            compressed_text = mlp(text_feat.unsqueeze(0)).squeeze(0)
            sims = features_t @ compressed_text

    sims_np = sims.numpy()
    threshold = np.quantile(sims_np, 1.0 - args.top)
    top_mask = sims_np >= threshold

    print(f"similarity — min: {sims_np.min():.3f}  max: {sims_np.max():.3f}  mean: {sims_np.mean():.3f}")
    print(f"top {args.top*100:.0f}% threshold: {threshold:.3f}  ({top_mask.sum():,} gaussians)")

    # color: top matches red, rest gray
    colors = np.full((len(xyz), 3), 0.6)
    colors[top_mask] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=f'query: "{args.query}"')


if __name__ == "__main__":
    main()
