"""
Lift SAM3 masks to 3D Gaussian space and build the scene graph.

Two-stage pipeline:
  1. Lifting  — for each Gaussian, find which object mask it falls under
               via 2D projection + plurality vote across frames.
  2. Graph    — per object: compute CLIP embedding from best-view crop,
               compute 3D bbox from assigned Gaussians, add spatial edges.

Usage:
    python pipeline/build_graph.py \
        --gaussians  desk_colmap_gaussian_15000.ply \
        --segments   pipeline/segments.json \
        --colmap     desk_colmap/sparse/0 \
        --scene      "splat_session 2" \
        --out        pipeline/scene_graph.json

Output (scene_graph.json) — see semantic-spatial-memory.md for full schema.

Design notes:
  - Plurality vote per Gaussian resolves overlapping masks gracefully.
    Gaussians with no mask votes are assigned to background.
  - Best frame = frame with largest mask area for that object.
    This gives the clearest crop for CLIP (most in-frame, least occluded).
  - CLIP embedding is computed from the masked crop only (background zeroed),
    not the full frame, so the embedding reflects the object not the scene.
  - Spatial edges: axis-aligned 'on top of' relation when object centroid Y
    is above another object's bbox top and within its XZ footprint.
    Intentionally simple — phase 2 will add richer relations.
  - structure nodes get embeddings too so they can anchor relational queries
    e.g. "cup on the desk".
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData
from pycocotools import mask as mask_utils


# minimum fraction of votes for a Gaussian to be assigned (not noise)
MIN_VOTE_FRACTION = 0.1

# CLIP model
CLIP_MODEL      = 'ViT-B-16'
CLIP_PRETRAINED = 'laion2b_s34b_b88k'


# ---------------------------------------------------------------------------
# COLMAP readers
# ---------------------------------------------------------------------------

def read_colmap_cameras(cameras_bin):
    """Returns dict: camera_id -> {w, h, fx, fy, cx, cy}"""
    cameras = {}
    with open(cameras_bin, 'rb') as f:
        n_cams = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_cams):
            cam_id, model_id = struct.unpack('<ii', f.read(8))
            w, h             = struct.unpack('<QQ', f.read(16))
            # read params based on model
            # model 1 = PINHOLE (fx, fy, cx, cy), model 0 = SIMPLE_PINHOLE (f, cx, cy)
            if model_id == 0:
                params = struct.unpack('<3d', f.read(24))
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            else:
                # PINHOLE or OPENCV etc — read 4 params, use first 4
                n_params = {1: 4, 2: 5, 3: 8, 4: 5, 5: 8}.get(model_id, 4)
                params   = struct.unpack(f'<{n_params}d', f.read(8 * n_params))
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            cameras[cam_id] = {'w': w, 'h': h, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return cameras


def read_colmap_images(images_bin):
    """
    Returns dict: image_name -> {cam_id, R (3x3), t (3,)}
    R and t define world-to-camera transform: x_cam = R @ x_world + t
    """
    images = {}
    with open(images_bin, 'rb') as f:
        n_imgs = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_imgs):
            img_id  = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
            tx, ty, tz      = struct.unpack('<3d', f.read(24))
            cam_id          = struct.unpack('<I', f.read(4))[0]
            # read null-terminated name
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode('utf-8')
            # skip 2D keypoints
            n_pts2d = struct.unpack('<Q', f.read(8))[0]
            f.read(n_pts2d * 24)
            # quaternion to rotation matrix
            R = quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            images[name] = {'cam_id': cam_id, 'R': R, 't': t}
    return images


def quat_to_rot(qw, qx, qy, qz):
    """Unit quaternion -> 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])
    return R


# ---------------------------------------------------------------------------
# Gaussian loading
# ---------------------------------------------------------------------------

def load_gaussians(ply_path):
    """Returns (N, 3) float32 xyz array."""
    ply = PlyData.read(str(ply_path))
    v   = ply['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_gaussians(xyz, cam, image_shape):
    """
    Project Gaussian centers into image coordinates.

    xyz:         (N, 3) world coords
    cam:         {'R': (3,3), 't': (3,), 'fx', 'fy', 'cx', 'cy', 'w', 'h'}
    image_shape: (H, W) — used for bounds check

    Returns:
        px (N,) int32 — pixel x (col)
        py (N,) int32 — pixel y (row)
        visible (N,) bool — within image bounds and in front of camera
    """
    R, t = cam['R'], cam['t']
    # world -> camera
    x_cam = (xyz @ R.T) + t           # (N, 3)
    z     = x_cam[:, 2]
    visible = z > 0.01                 # in front of camera

    u = cam['fx'] * (x_cam[:, 0] / (z + 1e-8)) + cam['cx']
    v = cam['fy'] * (x_cam[:, 1] / (z + 1e-8)) + cam['cy']

    H, W  = image_shape
    px    = np.round(u).astype(np.int32)
    py    = np.round(v).astype(np.int32)
    visible &= (px >= 0) & (px < W) & (py >= 0) & (py < H)

    return px, py, visible


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------

def decode_rle(rle):
    """Decode pycocotools RLE (compressed string or list) to bool H×W."""
    if isinstance(rle['counts'], str):
        rle_enc = {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
        return mask_utils.decode(rle_enc).astype(bool)
    # uncompressed list
    size = rle['size']
    flat = np.zeros(size[0] * size[1], dtype=bool)
    pos  = 0
    for i, c in enumerate(rle['counts']):
        if i % 2 == 1:
            flat[pos:pos + c] = True
        pos += c
    return flat.reshape(size)


# ---------------------------------------------------------------------------
# Lifting: assign Gaussians to objects via plurality vote
# ---------------------------------------------------------------------------

def lift_masks(xyz, segments, colmap_cameras, colmap_images, image_dir):
    """
    For each Gaussian, accumulate per-object vote counts across all frames.
    Plurality vote -> object assignment. No majority required.

    Returns:
        assignments: np.int32 (N,) — index into obj_ids, -1 = background
        obj_ids:     list of str   — object id for each index
        vote_counts: (N, n_objs)  — raw vote matrix (for diagnostics)
    """
    N       = len(xyz)
    obj_ids = list(segments['objects'].keys())
    n_objs  = len(obj_ids)
    obj_idx = {oid: i for i, oid in enumerate(obj_ids)}

    vote_counts = np.zeros((N, n_objs), dtype=np.int32)

    frame_names = list(colmap_images.keys())
    print(f'  {len(frame_names)} frames, {N:,} Gaussians, {n_objs} objects')

    for fi, frame_name in enumerate(frame_names):
        if fi % 50 == 0:
            print(f'  frame {fi}/{len(frame_names)}')

        img_meta = colmap_images[frame_name]
        cam_meta = colmap_cameras[img_meta['cam_id']]

        # build combined camera dict for projection
        cam = {**cam_meta, 'R': img_meta['R'], 't': img_meta['t']}

        # load image shape (need H, W)
        img_path = image_dir / frame_name
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            W_img, H_img = img.size
        image_shape = (H_img, W_img)

        px, py, visible = project_gaussians(xyz, cam, image_shape)
        if not visible.any():
            continue

        vis_px = px[visible]
        vis_py = py[visible]

        # accumulate votes for each object that has a mask in this frame
        for obj_id, obj in segments['objects'].items():
            if frame_name not in obj['frames']:
                continue
            rle  = obj['frames'][frame_name]['mask_rle']
            mask = decode_rle(rle)
            if mask.shape != (H_img, W_img):
                continue
            # which visible Gaussians fall inside this mask
            hit = mask[vis_py, vis_px]
            vote_counts[visible, obj_idx[obj_id]] += hit.astype(np.int32)

    # plurality vote
    total_votes = vote_counts.sum(axis=1)
    assignments = np.full(N, -1, dtype=np.int32)
    has_votes   = total_votes > 0
    best_obj    = np.argmax(vote_counts[has_votes], axis=1)
    best_frac   = vote_counts[has_votes][np.arange(best_obj.shape[0]), best_obj] / (
        total_votes[has_votes] + 1e-6
    )
    # only assign if plurality is confident enough
    confident              = best_frac >= MIN_VOTE_FRACTION
    assignments[has_votes] = np.where(confident, best_obj, -1)

    return assignments, obj_ids, vote_counts


# ---------------------------------------------------------------------------
# CLIP embedding from best-view crop
# ---------------------------------------------------------------------------

def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


def embed_object_crop(obj_id, obj, image_dir, clip_model, clip_preprocess, device):
    """
    Pick best frame (largest mask area), crop + mask it, embed with CLIP.
    Returns (embedding np float32 [512], best_frame_name).
    """
    best_frame = max(obj['frames'].items(), key=lambda kv: kv[1]['area_px'])
    frame_name, frame_data = best_frame

    img_path = image_dir / frame_name
    if not img_path.exists():
        return None, frame_name

    img  = np.array(Image.open(img_path).convert('RGB'))
    rle  = frame_data['mask_rle']
    mask = decode_rle(rle)
    H, W = img.shape[:2]
    if mask.shape != (H, W):
        return None, frame_name

    # zero out background — embedding reflects object, not scene context
    masked = img.copy()
    masked[~mask] = 0

    # tight crop to mask bbox
    x0, y0, x1, y1 = frame_data['bbox_px']
    # add small padding
    pad  = 8
    x0   = max(0, x0 - pad)
    y0   = max(0, y0 - pad)
    x1   = min(W, x1 + pad)
    y1   = min(H, y1 + pad)
    crop = masked[y0:y1, x0:x1]

    if crop.size == 0:
        return None, frame_name

    crop_pil = Image.fromarray(crop)
    tensor   = clip_preprocess(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0).cpu().numpy().tolist(), frame_name


# ---------------------------------------------------------------------------
# 3D bbox from assigned Gaussians
# ---------------------------------------------------------------------------

def compute_bbox(xyz, gaussian_ids):
    """Returns {'min', 'max', 'centroid'} dicts from assigned Gaussian positions."""
    if not gaussian_ids:
        return None
    pts = xyz[gaussian_ids]
    mn  = pts.min(axis=0).tolist()
    mx  = pts.max(axis=0).tolist()
    cen = pts.mean(axis=0).tolist()
    return {'min': mn, 'max': mx, 'centroid': cen}


# ---------------------------------------------------------------------------
# Spatial edge inference
# ---------------------------------------------------------------------------

def infer_edges(nodes):
    """
    Infer 'on top of' spatial relations between object nodes.
    Simple heuristic: object A is 'on top of' object B if:
      - A's centroid Y is above B's bbox top (Y_min in COLMAP convention
        depends on scene orientation — use centroid comparison instead)
      - A's centroid XZ is within B's XZ footprint

    Returns list of {source, target, relation} dicts.
    COLMAP uses right-hand coords. Y is up or down depending on capture.
    We compare centroids: if A.centroid[1] < B.centroid[1] (smaller Y = higher
    in typical COLMAP scenes captured looking forward), A is above B.
    This is a best-effort heuristic; adjust sign per scene if needed.
    """
    edges     = []
    obj_nodes = [(nid, n) for nid, n in nodes.items()
                 if n['type'] in ('object', 'structure') and n['bbox'] is not None]

    for i, (id_a, node_a) in enumerate(obj_nodes):
        for id_b, node_b in obj_nodes[i + 1:]:
            if id_a == id_b:
                continue
            ca = np.array(node_a['bbox']['centroid'])
            cb = np.array(node_b['bbox']['centroid'])
            bb_min = np.array(node_b['bbox']['min'])
            bb_max = np.array(node_b['bbox']['max'])

            # check XZ footprint overlap (indices 0, 2)
            in_x = bb_min[0] <= ca[0] <= bb_max[0]
            in_z = bb_min[2] <= ca[2] <= bb_max[2]
            if not (in_x and in_z):
                continue

            # check A is above B (lower Y index in COLMAP cam coords typically = higher)
            # use centroid distance as a proxy — skip if same height
            if abs(ca[1] - cb[1]) < 0.05:
                continue

            if ca[1] < cb[1]:
                edges.append({'source': id_a, 'target': id_b, 'relation': 'on top of'})
            else:
                edges.append({'source': id_b, 'target': id_a, 'relation': 'on top of'})

    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussians', required=True, help='PLY file of trained Gaussians')
    parser.add_argument('--segments',  required=True, help='segments.json from segment_masks.py')
    parser.add_argument('--colmap',    required=True, help='COLMAP sparse/0 directory')
    parser.add_argument('--scene',     required=True, help='scene directory (contains images/)')
    parser.add_argument('--out',       required=True, help='output scene_graph.json path')
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    colmap_dir = Path(args.colmap)
    image_dir  = Path(args.scene) / 'images'

    print('loading Gaussians...')
    xyz = load_gaussians(args.gaussians)
    print(f'  {len(xyz):,} Gaussians')
    print()

    print('loading segments...')
    with open(args.segments) as f:
        segments = json.load(f)
    n_obj    = sum(1 for o in segments['objects'].values() if o['type'] == 'object')
    n_struct = sum(1 for o in segments['objects'].values() if o['type'] == 'structure')
    print(f'  {n_obj} objects, {n_struct} structures')
    print()

    print('reading COLMAP...')
    colmap_cameras = read_colmap_cameras(colmap_dir / 'cameras.bin')
    colmap_images  = read_colmap_images(colmap_dir  / 'images.bin')
    print(f'  {len(colmap_cameras)} cameras, {len(colmap_images)} images')
    print()

    print('lifting masks to Gaussians...')
    assignments, obj_ids, vote_counts = lift_masks(
        xyz, segments, colmap_cameras, colmap_images, image_dir
    )
    assigned_obj = (assignments >= 0).sum()
    print(f'  {assigned_obj:,}/{len(xyz):,} Gaussians assigned ({100*assigned_obj/len(xyz):.1f}%)')
    print()

    print('loading CLIP...')
    clip_model, clip_preprocess, _ = load_clip(args.device)
    print()

    print('building scene graph nodes...')
    nodes = {}

    for obj_id, obj in segments['objects'].items():
        idx        = obj_ids.index(obj_id)
        gauss_ids  = list(np.where(assignments == idx)[0].astype(int))
        bbox       = compute_bbox(xyz, gauss_ids) if gauss_ids else None

        print(f'  {obj_id} ({obj["label"]}): {len(gauss_ids):,} Gaussians')

        emb, best_frame = embed_object_crop(
            obj_id, obj, image_dir, clip_model, clip_preprocess, args.device
        )
        if emb is None:
            print(f'    WARN: could not compute CLIP embedding')

        # best frame area for confidence proxy
        best_area = max(fd['area_px'] for fd in obj['frames'].values())

        nodes[obj_id] = {
            'label':          obj['label'],
            'type':           obj['type'],
            'clip_embedding': emb,
            'gaussian_ids':   gauss_ids,
            'bbox':           bbox,
            'best_frame':     best_frame,
            'mask_area_px':   best_area,
            'confidence':     None,  # SAM3 does not expose instance scores in current API
        }

    # background node — all unassigned Gaussians
    bg_ids = list(np.where(assignments == -1)[0].astype(int))
    nodes['background'] = {
        'label':          'background',
        'type':           'background',
        'clip_embedding': None,
        'gaussian_ids':   bg_ids,
        'bbox':           None,
        'best_frame':     None,
        'mask_area_px':   None,
        'confidence':     None,
    }
    print(f'  background: {len(bg_ids):,} Gaussians')
    print()

    print('inferring spatial edges...')
    edges = infer_edges(nodes)
    print(f'  {len(edges)} edges')
    print()

    graph = {'nodes': nodes, 'edges': edges}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(graph, f)

    print(f'saved: {out_path}')
    print(f'nodes: {len(nodes)}  edges: {len(edges)}')


if __name__ == '__main__':
    main()
