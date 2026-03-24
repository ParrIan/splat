"""
Train DINOv2->CLIP projection and bake features into Gaussian PLY.

No dependency on the gaussian-splatting repo — reads COLMAP binaries directly.

Usage (Colab):
    python pipeline/train_features.py \
        --zip       /drive/MyDrive/splat/desk_colmap.zip \
        --gaussians /drive/MyDrive/splat/desk_colmap_gaussian_15000.ply \
        --out_dir   /drive/MyDrive/splat \
        --stem      desk_colmap

All intermediate files written to /content/splat_work/.
"""

import argparse
import glob
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

DINO_DIM   = 768
CLIP_DIM   = 512
DINO_PATCH = 14
WORK_DIR   = Path('/content/splat_work')

dino_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id = struct.unpack('<ii', f.read(8))
            w, h = struct.unpack('<QQ', f.read(16))
            n_params = {0: 3, 1: 4, 2: 4, 3: 5}.get(model_id, 4)
            params = struct.unpack(f'<{n_params}d', f.read(8 * n_params))
            cameras[cam_id] = {'width': int(w), 'height': int(h), 'params': params, 'model': model_id}
    return cameras


def read_images_bin(path):
    images = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            image_id, = struct.unpack('<i', f.read(4))
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            camera_id, = struct.unpack('<i', f.read(4))
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            n_pts = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * n_pts)
            images.append({
                'image_id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': name.decode(),
            })
    return images


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,   1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw,   1 - 2*qx**2 - 2*qy**2],
    ])


def get_intrinsics(cam):
    p = cam['params']
    if cam['model'] == 0:   # SIMPLE_PINHOLE
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]  # PINHOLE


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
# Gaussian loading
# ---------------------------------------------------------------------------

def load_gaussians(ply_path):
    ply = PlyData.read(str(ply_path))
    v   = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    print(f'loaded {len(xyz):,} gaussians')
    return xyz


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_dino_features(dino_model, image_path, H, W, device):
    """Returns (H, W, 768) L2-normalized DINOv2 patch features."""
    dino_h = max((H // DINO_PATCH) * DINO_PATCH, DINO_PATCH)
    dino_w = max((W // DINO_PATCH) * DINO_PATCH, DINO_PATCH)
    img = Image.open(image_path).convert('RGB').resize((dino_w, dino_h))
    x = dino_transform(img).unsqueeze(0).to(device)
    feats = dino_model.get_intermediate_layers(x, n=1)[0]
    grid_h = dino_h // DINO_PATCH
    grid_w = dino_w // DINO_PATCH
    feats = feats[0].reshape(grid_h, grid_w, DINO_DIM)
    feats = feats.permute(2, 0, 1).unsqueeze(0)
    feats = torch.nn.functional.interpolate(
        feats, size=(H, W), mode='bilinear', align_corners=False
    )
    feats = feats[0].permute(1, 2, 0)
    return torch.nn.functional.normalize(feats, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Gaussian aggregation
# ---------------------------------------------------------------------------

def aggregate_dino_features(xyz, cameras_bin, images_bin, image_dir, dino_model, device):
    """
    Inverse-depth weighted aggregation of DINOv2 features per Gaussian.
    Also tracks best camera (lowest depth) per Gaussian for CLIP supervision.

    Returns:
        raw_dino:    (N, 768) aggregated DINOv2 features, L2-normalized
        valid:       (N,) bool mask — Gaussians seen by at least one camera
        best_cam:    (N,) int — index into images_bin of best camera per Gaussian
        best_px:     (N, 2) int — [px, py] in best camera
    """
    N = len(xyz)
    dino_sum   = np.zeros((N, DINO_DIM), dtype=np.float32)
    weight_sum = np.zeros(N, dtype=np.float32)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    best_cam   = np.full(N, -1, dtype=np.int32)
    best_px    = np.zeros((N, 2), dtype=np.int32)

    for cam_idx, img_entry in enumerate(tqdm(images_bin, desc='aggregating DINOv2')):
        cam = cameras_bin[img_entry['camera_id']]
        W, H = cam['width'], cam['height']
        img_path = image_dir / img_entry['name']
        if not img_path.exists():
            print(f'  missing: {img_path}')
            continue

        R = qvec_to_rotmat(img_entry['qvec'])
        t = img_entry['tvec']
        fx, fy, cx, cy = get_intrinsics(cam)

        pts_cam = (R @ xyz.T).T + t       # (N, 3)
        depth   = pts_cam[:, 2]
        valid_d = depth > 0.01

        px_f = (fx * pts_cam[valid_d, 0] / pts_cam[valid_d, 2]) + cx
        py_f = (fy * pts_cam[valid_d, 1] / pts_cam[valid_d, 2]) + cy
        px   = px_f.astype(np.int32)
        py   = py_f.astype(np.int32)

        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        global_idx = np.where(valid_d)[0][in_bounds]
        vis_px  = px[in_bounds]
        vis_py  = py[in_bounds]
        vis_d   = depth[valid_d][in_bounds]

        if len(global_idx) == 0:
            continue

        dino_feat = extract_dino_features(dino_model, img_path, H, W, device)
        feats = dino_feat[vis_py, vis_px].numpy()   # (M, 768)
        w = 1.0 / vis_d.clip(0.01)

        np.add.at(dino_sum,   global_idx, feats * w[:, None])
        np.add.at(weight_sum, global_idx, w)

        improved = vis_d < best_depth[global_idx]
        upd_idx  = global_idx[improved]
        best_depth[upd_idx] = vis_d[improved]
        best_cam[upd_idx]   = cam_idx
        best_px[upd_idx]    = np.stack([vis_px[improved], vis_py[improved]], axis=1)

    valid    = weight_sum > 0
    raw_dino = np.zeros((N, DINO_DIM), dtype=np.float32)
    raw_dino[valid] = dino_sum[valid] / weight_sum[valid, None]
    norms = np.linalg.norm(raw_dino[valid], axis=1, keepdims=True).clip(1e-6)
    raw_dino[valid] /= norms

    print(f'aggregated for {valid.sum():,} / {N:,} gaussians')
    return raw_dino, valid, best_cam, best_px


# ---------------------------------------------------------------------------
# Per-Gaussian CLIP supervision
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_gaussian_clip_targets(xyz, valid, best_cam, best_px, images_bin, cameras_bin,
                                 image_dir, clip_model, clip_prep, device,
                                 crop_size=64, batch_size=256):
    """
    For each valid Gaussian, crop a window around its best-camera projection
    and get the CLIP image embedding of that crop.

    Returns:
        dino_inputs:  (M, 768) raw_dino rows for valid Gaussians with a best camera
        clip_targets: (M, 512) CLIP embeddings of per-Gaussian crops
        pair_idx:     (M,) indices into the full Gaussian array
    """
    valid_idx  = np.where(valid)[0]
    has_cam    = best_cam[valid_idx] >= 0
    pair_idx   = valid_idx[has_cam]
    pair_cam   = best_cam[pair_idx]
    pair_px    = best_px[pair_idx]   # (M, 2)

    print(f'building CLIP targets for {len(pair_idx):,} gaussians...')

    # preload images grouped by camera to avoid re-opening
    cam_to_pairs = {}
    for i, ci in enumerate(pair_cam):
        cam_to_pairs.setdefault(ci, []).append(i)

    clip_targets = np.zeros((len(pair_idx), CLIP_DIM), dtype=np.float32)

    for cam_idx, pair_indices in tqdm(cam_to_pairs.items(), desc='CLIP crops'):
        img_entry = images_bin[cam_idx]
        cam = cameras_bin[img_entry['camera_id']]
        W, H = cam['width'], cam['height']
        img_path = image_dir / img_entry['name']
        if not img_path.exists():
            continue

        img_pil = Image.open(img_path).convert('RGB')
        half = crop_size // 2

        crops = []
        valid_pairs = []
        for i in pair_indices:
            px, py = pair_px[i]
            x0 = max(0, px - half); x1 = min(W, px + half)
            y0 = max(0, py - half); y1 = min(H, py + half)
            if (x1 - x0) < 16 or (y1 - y0) < 16:
                continue
            crops.append(clip_prep(img_pil.crop((x0, y0, x1, y1))))
            valid_pairs.append(i)

        if not crops:
            continue

        for b in range(0, len(crops), batch_size):
            batch = torch.stack(crops[b:b+batch_size]).to(device)
            embs  = clip_model.encode_image(batch)
            embs  = torch.nn.functional.normalize(embs, dim=-1).cpu().numpy()
            for j, i in enumerate(valid_pairs[b:b+batch_size]):
                clip_targets[i] = embs[j]

    # filter out any pairs where CLIP target is still zero (missing images)
    nonzero = np.linalg.norm(clip_targets, axis=1) > 0.5
    pair_idx     = pair_idx[nonzero]
    clip_targets = clip_targets[nonzero]

    print(f'clip targets built: {len(pair_idx):,} pairs')
    return pair_idx, clip_targets


# ---------------------------------------------------------------------------
# Projection model + training
# ---------------------------------------------------------------------------

class DinoToClipProjection(nn.Module):
    def __init__(self, dino_dim=768, clip_dim=512):
        super().__init__()
        self.proj = nn.Linear(dino_dim, clip_dim, bias=False)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


def train_projection(raw_dino, pair_idx, clip_targets, device,
                     epochs=300, batch=512, lr=1e-3, uniformity_weight=0.5):
    dino_inputs  = torch.from_numpy(raw_dino[pair_idx]).to(device)
    clip_targets_t = torch.from_numpy(clip_targets).to(device)

    projection = DinoToClipProjection(DINO_DIM, CLIP_DIM).to(device)
    optimizer  = torch.optim.Adam(projection.parameters(), lr=lr)

    for epoch in range(epochs):
        perm       = torch.randperm(len(dino_inputs), device=device)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(dino_inputs), batch):
            idx        = perm[i:i+batch]
            proj_batch = projection(dino_inputs[idx])
            tgt_batch  = clip_targets_t[idx]

            align_loss = 1.0 - (proj_batch * tgt_batch).sum(dim=-1).mean()

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
    ply    = PlyData.read(str(gaussian_ply_path))
    vertex = ply['vertex']
    vertex_data = {prop.name: vertex[prop.name] for prop in vertex.properties}

    feat_np = final_features
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
    parser.add_argument('--zip',        required=True,  help='colmap session zip')
    parser.add_argument('--gaussians',  required=True,  help='trained gaussian PLY')
    parser.add_argument('--out_dir',    required=True,  help='output directory (Drive)')
    parser.add_argument('--stem',       required=True,  help='output filename stem')
    parser.add_argument('--crop_size',  type=int,   default=64)
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch',      type=int,   default=512)
    parser.add_argument('--uniformity', type=float, default=0.5)
    parser.add_argument('--device',     default='cuda')
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
    scene_dir  = Path(candidates[0]).parent if candidates else scene_dir
    sparse_dir = scene_dir / 'sparse' / '0'
    image_dir  = scene_dir / 'images'
    print(f'scene: {scene_dir}')

    cameras_bin = read_cameras_bin(sparse_dir / 'cameras.bin')
    images_bin  = read_images_bin(sparse_dir / 'images.bin')
    print(f'cameras: {len(cameras_bin)},  images: {len(images_bin)}')

    xyz = load_gaussians(args.gaussians)
    N   = len(xyz)

    dino_model, clip_model, clip_prep = load_models(device)

    raw_dino, valid, best_cam, best_px = aggregate_dino_features(
        xyz, cameras_bin, images_bin, image_dir, dino_model, device
    )

    pair_idx, clip_targets = build_gaussian_clip_targets(
        xyz, valid, best_cam, best_px, images_bin, cameras_bin,
        image_dir, clip_model, clip_prep, device,
        crop_size=args.crop_size,
    )

    projection = train_projection(
        raw_dino, pair_idx, clip_targets, device,
        epochs=args.epochs, batch=args.batch, uniformity_weight=args.uniformity
    )

    # apply projection to all valid gaussians
    projection.eval()
    final_features = np.zeros((N, CLIP_DIM), dtype=np.float32)
    valid_idx  = np.where(valid)[0]
    valid_dino = torch.from_numpy(raw_dino[valid_idx]).to(device)

    with torch.no_grad():
        for i in range(0, len(valid_dino), 8192):
            out = projection(valid_dino[i:i+8192]).cpu().numpy()
            final_features[valid_idx[i:i+8192]] = out

    print(f'final features: {final_features.shape}')

    feature_ply = WORK_DIR / 'point_cloud_with_features.ply'
    save_feature_ply(args.gaussians, final_features, feature_ply, N)

    shutil.copy(feature_ply, out_dir / f'{args.stem}_features.ply')
    proj_pt = WORK_DIR / 'projection.pt'
    torch.save(projection.state_dict(), proj_pt)
    shutil.copy(proj_pt, out_dir / f'{args.stem}_projection.pt')

    print(f'\noutputs saved to {out_dir}:')
    print(f'  {args.stem}_features.ply')
    print(f'  {args.stem}_projection.pt')


if __name__ == '__main__':
    main()
