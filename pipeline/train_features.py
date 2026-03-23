"""
Train DINOv2->CLIP projection and bake features into Gaussian PLY.

Usage (Colab):
    python pipeline/train_features.py \
        --zip      /drive/MyDrive/splat/desk_colmap.zip \
        --gaussians /drive/MyDrive/splat/desk_colmap_gaussian_15000.ply \
        --out_dir  /drive/MyDrive/splat \
        --stem     desk_colmap

All intermediate files written to /content/splat_work/.
"""

import argparse
import glob
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

DINO_DIM   = 768
CLIP_DIM   = 512
DINO_PATCH = 14
WORK_DIR   = Path('/content/splat_work')


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_models(device):
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    dino.eval()

    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', device=device
    )
    clip_model.eval()

    print('dino + clip loaded')
    return dino, clip_model, clip_prep


# ---------------------------------------------------------------------------
# Scene loading (gaussian-splatting repo)
# ---------------------------------------------------------------------------

def load_scene(scene_dir, output_dir, gaussian_ply, iterations=15000):
    sys.path.insert(0, '/content/gaussian-splatting')
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos
    from arguments import ModelParams

    ply_dest = Path(output_dir) / f'point_cloud/iteration_{iterations}/point_cloud.ply'
    ply_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(gaussian_ply, ply_dest)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_dest))
    N = gaussians.get_xyz.shape[0]
    print(f'loaded {N:,} gaussians')

    parser = argparse.ArgumentParser()
    ModelParams(parser)
    args = parser.parse_args(['-s', str(scene_dir), '-m', str(output_dir)])

    scene_info = sceneLoadTypeCallbacks['Colmap'](str(scene_dir), 'images', '', False, False)
    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args, False, False)
    print(f'loaded {len(cameras)} training cameras')

    return gaussians, cameras, N, ply_dest


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

dino_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_dino_features(dino_model, image_path, target_h, target_w, device):
    """Returns (H, W, 768) L2-normalized DINOv2 patch features."""
    dino_h = max((target_h // DINO_PATCH) * DINO_PATCH, DINO_PATCH)
    dino_w = max((target_w // DINO_PATCH) * DINO_PATCH, DINO_PATCH)

    img = Image.open(image_path).convert('RGB').resize((dino_w, dino_h))
    x = dino_transform(img).unsqueeze(0).to(device)

    feats = dino_model.get_intermediate_layers(x, n=1)[0]
    grid_h = dino_h // DINO_PATCH
    grid_w = dino_w // DINO_PATCH
    feats = feats[0].reshape(grid_h, grid_w, DINO_DIM)
    feats = feats.permute(2, 0, 1).unsqueeze(0)
    feats = torch.nn.functional.interpolate(
        feats, size=(target_h, target_w), mode='bilinear', align_corners=False
    )
    feats = feats[0].permute(1, 2, 0)
    return torch.nn.functional.normalize(feats, dim=-1).cpu()


@torch.no_grad()
def get_clip_image_embedding(clip_model, clip_prep, img_pil, device):
    """Returns (512,) normalized CLIP image embedding."""
    x = clip_prep(img_pil).unsqueeze(0).to(device)
    emb = clip_model.encode_image(x)
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Gaussian feature aggregation
# ---------------------------------------------------------------------------

def aggregate_dino_features(gaussians, cameras, image_dir, dino_model, device, N):
    gaussians_xyz = gaussians.get_xyz.detach()

    dino_sum   = torch.zeros(N, DINO_DIM, dtype=torch.float32)
    weight_sum = torch.zeros(N, dtype=torch.float32)

    for cam in tqdm(cameras, desc='aggregating DINOv2 features'):
        H, W = cam.image_height, cam.image_width
        img_path = image_dir / cam.image_name
        if not img_path.exists():
            print(f'  missing: {img_path}')
            continue

        dino_feat = extract_dino_features(dino_model, img_path, H, W, device)

        xyz_h    = torch.cat([gaussians_xyz, torch.ones(N, 1, device=device)], dim=1)
        view_mat = cam.world_view_transform.to(device)
        proj_mat = cam.full_proj_transform.to(device)

        pts_view       = xyz_h @ view_mat
        pts_clip_space = xyz_h @ proj_mat

        w   = pts_clip_space[:, 3:4].clamp(min=1e-6)
        ndc = pts_clip_space[:, :2] / w

        px    = ((ndc[:, 0] + 1) * 0.5 * W).long()
        py    = ((1 - (ndc[:, 1] + 1) * 0.5) * H).long()
        depth = pts_view[:, 2]

        visible = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (depth > 0)
        vis_idx = visible.nonzero(as_tuple=True)[0]
        if vis_idx.numel() == 0:
            continue

        vis_px    = px[vis_idx].cpu()
        vis_py    = py[vis_idx].cpu()
        vis_depth = depth[vis_idx].cpu()
        weights   = 1.0 / vis_depth.clamp(min=0.01)
        feats     = dino_feat[vis_py, vis_px]

        dino_sum.index_add_(0, vis_idx.cpu(), feats * weights.unsqueeze(1))
        weight_sum.index_add_(0, vis_idx.cpu(), weights)

    valid    = weight_sum > 0
    raw_dino = torch.zeros(N, DINO_DIM)
    raw_dino[valid] = dino_sum[valid] / weight_sum[valid].unsqueeze(1)
    raw_dino = torch.nn.functional.normalize(raw_dino, dim=-1)

    print(f'aggregated dino features for {valid.sum().item():,} / {N:,} gaussians')
    return raw_dino, valid


# ---------------------------------------------------------------------------
# Projection model
# ---------------------------------------------------------------------------

class DinoToClipProjection(nn.Module):
    def __init__(self, dino_dim=768, clip_dim=512):
        super().__init__()
        self.proj = nn.Linear(dino_dim, clip_dim, bias=False)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


# ---------------------------------------------------------------------------
# Build supervision pairs
# ---------------------------------------------------------------------------

def build_region_pairs(cameras, image_dir, dino_model, clip_model, clip_prep,
                       device, n_regions=8):
    all_dino = []
    all_clip = []

    for cam in tqdm(cameras, desc='building region pairs'):
        H, W = cam.image_height, cam.image_width
        img_path = image_dir / cam.image_name
        if not img_path.exists():
            continue

        img_pil = Image.open(img_path).convert('RGB')

        dino_h = max((H // DINO_PATCH) * DINO_PATCH, DINO_PATCH)
        dino_w = max((W // DINO_PATCH) * DINO_PATCH, DINO_PATCH)
        img_resized = img_pil.resize((dino_w, dino_h))
        x = dino_transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            patch_feats = dino_model.get_intermediate_layers(x, n=1)[0][0]

        grid_h = dino_h // DINO_PATCH
        grid_w = dino_w // DINO_PATCH
        patch_feats_np = patch_feats.cpu().numpy()

        k = min(n_regions, patch_feats_np.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0)
        labels = kmeans.fit_predict(patch_feats_np).reshape(grid_h, grid_w)

        scale_y = H / grid_h
        scale_x = W / grid_w

        for region_id in range(k):
            mask = labels == region_id
            if mask.sum() < 2:
                continue

            ys, xs = np.where(mask)
            patch_idx   = ys * grid_w + xs
            region_dino = torch.from_numpy(patch_feats_np[patch_idx]).mean(0)
            region_dino = torch.nn.functional.normalize(region_dino, dim=0)

            y0 = int(ys.min() * scale_y)
            y1 = int((ys.max() + 1) * scale_y)
            x0 = int(xs.min() * scale_x)
            x1 = int((xs.max() + 1) * scale_x)

            if (y1 - y0) < 16 or (x1 - x0) < 16:
                continue

            crop     = img_pil.crop((x0, y0, x1, y1))
            clip_emb = get_clip_image_embedding(clip_model, clip_prep, crop, device)

            all_dino.append(region_dino)
            all_clip.append(clip_emb)

    print(f'region pairs: {len(all_dino):,}')
    return torch.stack(all_dino), torch.stack(all_clip)


# ---------------------------------------------------------------------------
# Projection training
# ---------------------------------------------------------------------------

def train_projection(dino_pairs, clip_targets, device,
                     epochs=300, batch=256, lr=1e-3, uniformity_weight=0.5):
    projection = DinoToClipProjection(DINO_DIM, CLIP_DIM).to(device)
    optimizer  = torch.optim.Adam(projection.parameters(), lr=lr)

    dino_pairs   = dino_pairs.to(device)
    clip_targets = clip_targets.to(device)

    for epoch in range(epochs):
        perm       = torch.randperm(len(dino_pairs))
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(dino_pairs), batch):
            idx        = perm[i:i+batch]
            proj_batch = projection(dino_pairs[idx])
            tgt_batch  = clip_targets[idx]

            align_loss = 1.0 - (proj_batch * tgt_batch).sum(dim=-1).mean()

            # uniformity loss: push projected features apart (Wang & Isola 2020)
            if proj_batch.shape[0] > 1:
                sq_dists     = torch.pdist(proj_batch, p=2).pow(2)
                uniform_loss = sq_dists.mul(-2).exp().mean().log()
            else:
                uniform_loss = torch.tensor(0.0, device=device)

            loss = align_loss + uniformity_weight * uniform_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 50 == 0:
            print(f'epoch {epoch+1}/{epochs}  loss: {total_loss/n_batches:.4f}')

    print('projection training done')
    return projection


# ---------------------------------------------------------------------------
# Save feature PLY
# ---------------------------------------------------------------------------

def save_feature_ply(gaussian_ply_path, final_features, out_path, N):
    ply = PlyData.read(str(gaussian_ply_path))
    vertex = ply['vertex']
    vertex_data = {prop.name: vertex[prop.name] for prop in vertex.properties}

    feat_np = final_features.numpy()
    for i in range(CLIP_DIM):
        vertex_data[f'f_clip_{i}'] = feat_np[:, i].astype(np.float32)

    dtype = [(prop.name, vertex[prop.name].dtype) for prop in vertex.properties]
    dtype += [(f'f_clip_{i}', np.float32) for i in range(CLIP_DIM)]

    new_vertex = np.empty(N, dtype=dtype)
    for name, _ in dtype:
        new_vertex[name] = vertex_data[name]

    PlyData([PlyElement.describe(new_vertex, 'vertex')]).write(str(out_path))
    print(f'saved feature PLY: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip',       required=True,  help='path to colmap session zip')
    parser.add_argument('--gaussians', required=True,  help='path to trained gaussian PLY')
    parser.add_argument('--out_dir',   required=True,  help='output directory (Drive)')
    parser.add_argument('--stem',      required=True,  help='output filename stem')
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--n_regions',  type=int, default=8)
    parser.add_argument('--epochs',     type=int, default=300)
    parser.add_argument('--uniformity', type=float, default=0.5,
                        help='weight for uniformity loss (0 = disabled)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # unpack scene zip
    scene_dir = WORK_DIR / 'scene'
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(scene_dir)
    candidates = glob.glob(str(scene_dir / '**' / 'sparse'), recursive=True)
    if candidates:
        scene_dir = Path(candidates[0]).parent
    print(f'scene dir: {scene_dir}')

    output_dir = WORK_DIR / 'gs_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussians, cameras, N, ply_path = load_scene(
        scene_dir, output_dir, args.gaussians, args.iterations
    )
    image_dir = scene_dir / 'images'

    dino_model, clip_model, clip_prep = load_models(device)

    raw_dino, valid = aggregate_dino_features(
        gaussians, cameras, image_dir, dino_model, device, N
    )

    dino_pairs, clip_targets = build_region_pairs(
        cameras, image_dir, dino_model, clip_model, clip_prep,
        device, n_regions=args.n_regions
    )

    projection = train_projection(
        dino_pairs, clip_targets, device,
        epochs=args.epochs, uniformity_weight=args.uniformity
    )

    # project valid gaussians
    projection.eval()
    final_features = torch.zeros(N, CLIP_DIM)
    valid_dino = raw_dino[valid].to(device)
    valid_idx  = valid.nonzero(as_tuple=True)[0]

    with torch.no_grad():
        for i in range(0, valid_dino.shape[0], 8192):
            final_features[valid_idx[i:i+8192]] = projection(valid_dino[i:i+8192]).cpu()

    print(f'final features: {final_features.shape}')

    feature_ply = WORK_DIR / 'point_cloud_with_features.ply'
    save_feature_ply(ply_path, final_features, feature_ply, N)

    # copy outputs to Drive
    shutil.copy(feature_ply, out_dir / f'{args.stem}_features.ply')
    proj_pt = WORK_DIR / 'projection.pt'
    torch.save(projection.state_dict(), proj_pt)
    shutil.copy(proj_pt, out_dir / f'{args.stem}_projection.pt')

    print(f'\noutputs saved to {out_dir}:')
    print(f'  {args.stem}_features.ply')
    print(f'  {args.stem}_projection.pt')


if __name__ == '__main__':
    main()
