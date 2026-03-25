"""
Run Grounded-SAM2 concept segmentation on training frames.

Uses Grounding DINO to detect objects by noun phrase on each frame,
then SAM2 to segment and track them across the video with consistent IDs.

This is a drop-in replacement for the SAM3 version — same nouns.json input,
same segments.json output schema. When SAM3 access is available, swap this
file for the SAM3 version with no downstream changes.

Architecture:
  - Grounding DINO: open-vocab detector, noun phrase -> bounding boxes on each frame
  - SAM2 video predictor: box prompt on first detection frame -> mask + tracking

One SAM2 session per noun. Within each session, each detected instance gets
a unique object ID that is consistent across all subsequent frames via tracking.

Usage (Colab):
    python pipeline/segment_masks.py \
        --scene /content/splat/splat_session \
        --nouns pipeline/nouns.json \
        --out   pipeline/segments.json

Install:
    pip install groundingdino-py
    pip install git+https://github.com/facebookresearch/sam2.git
    # download weights to /content/weights/:
    #   groundingdino_swint_ogc.pth
    #   sam2.1_hiera_large.pt
    # clone GroundingDINO for config file:
    #   git clone https://github.com/IDEA-Research/GroundingDINO /content/GroundingDINO

nouns.json schema:
    {
      "objects":    ["cup", "keyboard", "monitor"],
      "structures": ["desk", "wall", "floor"]
    }

Output (segments.json):
    {
      "objects": {
        "<noun_slug>_<instance_id>": {
          "label":  str,
          "type":   "object" | "structure",
          "frames": {
            "<frame_name>": {
              "mask_rle":  {"counts": str, "size": [H, W]},
              "bbox_px":   [x0, y0, x1, y1],
              "area_px":   int
            }
          }
        }
      },
      "nouns":        {"objects": [...], "structures": [...]},
      "missed_nouns": [...]
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils


# Grounding DINO detection thresholds
GDINO_BOX_THRESHOLD  = 0.35
GDINO_TEXT_THRESHOLD = 0.25

# SAM2 model config — must match the checkpoint
SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sorted_frame_names(image_dir):
    exts  = {'.jpg', '.jpeg', '.png'}
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not names:
        raise RuntimeError(f'no images found in {image_dir}')
    return names


def mask_to_rle(mask_np):
    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def bbox_from_mask(mask_np):
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [x0, y0, x1, y1]


# ---------------------------------------------------------------------------
# Grounding DINO
# ---------------------------------------------------------------------------

def load_gdino(ckpt_path, config_path, device):
    from groundingdino.util.inference import load_model
    model = load_model(config_path, ckpt_path)
    return model.to(device).eval()


def detect_noun(gdino, image_pil, noun, device):
    """
    Detect all instances of noun in image_pil.
    Returns boxes (N, 4) in x0,y0,x1,y1 pixel coords and scores (N,).
    """
    from groundingdino.util.inference import predict
    from groundingdino.util import box_ops
    import torchvision.transforms.functional as TF

    W, H = image_pil.size
    image_tensor = TF.to_tensor(image_pil).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    with torch.no_grad():
        boxes, scores, _ = predict(
            model=gdino,
            image=image_tensor,
            caption=noun,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            device=device,
        )

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # convert from normalized cx,cy,w,h to pixel x0,y0,x1,y1
    boxes_px = box_ops.box_cxcywh_to_xyxy(boxes)
    boxes_px = boxes_px * torch.tensor([W, H, W, H], device=device, dtype=torch.float32)
    return boxes_px.cpu().numpy(), scores.cpu().numpy()


# ---------------------------------------------------------------------------
# SAM2 segmentation and tracking
# ---------------------------------------------------------------------------

def load_sam2(ckpt_path, device):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(SAM2_CONFIG, ckpt_path, device=device)


def run_noun_sam2(sam2, gdino, image_dir, frame_names, noun, device):
    """
    Detect noun with Grounding DINO, initialize SAM2 on first detection frame,
    propagate masks across all frames.

    Returns {instance_id: {frame_name: {mask_rle, bbox_px, area_px}}}
    """
    # find first frame with detections
    init_frame_idx = None
    init_boxes     = None

    for fi, frame_name in enumerate(frame_names):
        img = Image.open(image_dir / frame_name).convert('RGB')
        boxes, scores = detect_noun(gdino, img, noun, device)
        if len(boxes) > 0:
            init_frame_idx = fi
            init_boxes     = boxes
            print(f'    first detection at frame {fi} ({frame_name}): {len(boxes)} instance(s)')
            break

    if init_frame_idx is None:
        return {}

    results = {}

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = sam2.init_state(video_path=str(image_dir))
        sam2.reset_state(state)

        # one prompt per detected box, each gets its own object ID
        for inst_idx, box in enumerate(init_boxes):
            sam2.add_new_prompts(
                inference_state=state,
                frame_idx=init_frame_idx,
                obj_id=inst_idx,
                boxes=torch.tensor(box, dtype=torch.float32, device=device).unsqueeze(0),
            )

        for frame_idx, obj_ids, mask_logits in sam2.propagate_in_video(state):
            frame_name = frame_names[frame_idx]
            for i, obj_id in enumerate(obj_ids):
                mask = (mask_logits[i, 0] > 0).cpu().numpy()
                if not mask.any():
                    continue
                if obj_id not in results:
                    results[obj_id] = {}
                results[obj_id][frame_name] = {
                    'mask_rle': mask_to_rle(mask),
                    'bbox_px':  bbox_from_mask(mask),
                    'area_px':  int(mask.sum()),
                }

        sam2.reset_state(state)

    return results


# ---------------------------------------------------------------------------
# Main segmentation loop
# ---------------------------------------------------------------------------

def build_segments(sam2, gdino, image_dir, frame_names, nouns, device):
    noun_to_type = {}
    for label in nouns.get('objects', []):
        noun_to_type[label] = 'object'
    for label in nouns.get('structures', []):
        noun_to_type[label] = 'structure'

    all_nouns    = list(noun_to_type.keys())
    objects      = {}
    missed_nouns = []

    print(f'  {len(all_nouns)} nouns: {all_nouns}')

    for noun in all_nouns:
        print(f'  segmenting "{noun}"...')
        instances = run_noun_sam2(sam2, gdino, image_dir, frame_names, noun, device)

        if not instances:
            print(f'    WARN: no instances found')
            missed_nouns.append(noun)
            continue

        slug  = noun.replace(' ', '_')
        added = 0
        for inst_id, frame_data in instances.items():
            if frame_data:
                objects[f'{slug}_{inst_id}'] = {
                    'label':  noun,
                    'type':   noun_to_type[noun],
                    'frames': frame_data,
                }
                added += 1

        if added == 0:
            print(f'    WARN: all instances had empty frame data')
            missed_nouns.append(noun)
        else:
            print(f'    {added} instance(s)')

    return objects, missed_nouns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',      required=True)
    parser.add_argument('--nouns',      required=True)
    parser.add_argument('--out',        required=True)
    parser.add_argument('--gdino_ckpt', default='/content/weights/groundingdino_swint_ogc.pth')
    parser.add_argument('--gdino_cfg',  default='/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--sam2_ckpt',  default='/content/weights/sam2.1_hiera_large.pt')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    image_dir = scene_dir / 'images'
    if not image_dir.exists():
        raise RuntimeError(f'images dir not found: {image_dir}')

    with open(args.nouns) as f:
        nouns = json.load(f)

    print(f'scene: {scene_dir}')
    print(f'nouns: {nouns}')
    print()

    frame_names = sorted_frame_names(image_dir)
    if args.max_frames:
        frame_names = frame_names[:args.max_frames]
    print(f'{len(frame_names)} frames  ({frame_names[0]} ... {frame_names[-1]})')
    print()

    print('loading Grounding DINO...')
    gdino = load_gdino(args.gdino_ckpt, args.gdino_cfg, args.device)
    print()

    print('loading SAM2...')
    sam2 = load_sam2(args.sam2_ckpt, args.device)
    print()

    print('running segmentation...')
    objects, missed_nouns = build_segments(
        sam2, gdino, image_dir, frame_names, nouns, args.device
    )
    print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'objects':      objects,
            'nouns':        nouns,
            'missed_nouns': missed_nouns,
        }, f)

    n_obj    = sum(1 for o in objects.values() if o['type'] == 'object')
    n_struct = sum(1 for o in objects.values() if o['type'] == 'structure')
    print(f'objects:    {n_obj}')
    print(f'structures: {n_struct}')
    print(f'missed:     {missed_nouns}')
    print(f'saved:      {out_path}')


if __name__ == '__main__':
    main()
