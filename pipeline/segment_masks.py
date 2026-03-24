"""
Run SAM3 concept segmentation on training frames.

Takes a manually-provided noun list and runs SAM3 across all frames,
producing segments.json with RLE masks and consistent object IDs.

SAM3 API (from facebookresearch/sam3):
  - build_sam3_video_predictor() — loads model, downloads checkpoint from HuggingFace
  - predictor.handle_request(type="start_session", resource_path=<dir or mp4>)
  - predictor.handle_request(type="add_prompt", session_id=..., frame_index=0, text=<noun>)
  - predictor.handle_stream_request(type="propagate_in_video", session_id=...)
      yields per-frame responses with out_obj_ids, out_binary_masks, out_probs

Object IDs are assigned by SAM3's detector/tracker — globally unique integers
per session. We namespace them as "<noun_slug>_<sam3_obj_id>" in output.

Usage (Colab):
    python pipeline/segment_masks.py \
        --scene  "splat_session 2" \
        --nouns  pipeline/nouns.json \
        --out    pipeline/segments.json

nouns.json schema:
    {
      "objects":    ["cup", "keyboard", "monitor"],
      "structures": ["desk", "wall", "floor"]
    }

Output (segments.json):
    {
      "objects": {
        "<object_id>": {
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
from pycocotools import mask as mask_utils


def mask_to_rle(mask_np):
    """Encode binary H x W numpy bool array to pycocotools compressed RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def bbox_from_mask(mask_np):
    """Returns [x0, y0, x1, y1] tight bounding box of a binary mask."""
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y0, y1 = int(np.where(rows)[0][[0, -1]])
    x0, x1 = int(np.where(cols)[0][[0, -1]])
    return [x0, y0, x1, y1]


def sorted_frame_names(image_dir):
    """Returns sorted list of image filenames in image_dir."""
    exts  = {'.jpg', '.jpeg', '.png'}
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not names:
        raise RuntimeError(f'no images found in {image_dir}')
    return names


def run_noun(predictor, image_dir, frame_names, noun):
    """
    Run SAM3 for a single noun phrase across all frames.

    Returns dict:
        {obj_id_str: {frame_name: {mask_rle, bbox_px, area_px}}}
    or empty dict if no instances detected.
    """
    # SAM3 takes a directory of images or an mp4 path
    response   = predictor.handle_request(dict(
        type='start_session',
        resource_path=str(image_dir),
    ))
    session_id = response['session_id']

    try:
        # prompt on frame 0 — noun phrase is global across all frames
        prompt_resp = predictor.handle_request(dict(
            type='add_prompt',
            session_id=session_id,
            frame_index=0,
            text=noun,
        ))

        if not prompt_resp.get('outputs') or len(prompt_resp['outputs'].get('out_obj_ids', [])) == 0:
            return {}

        # propagate across entire video
        results = {}  # sam3_obj_id -> {frame_name -> {mask, score}}

        for response in predictor.handle_stream_request(dict(
            type='propagate_in_video',
            session_id=session_id,
        )):
            frame_idx = response['frame_index']
            outputs   = response['outputs']
            if frame_idx >= len(frame_names):
                continue
            frame_name = frame_names[frame_idx]

            obj_ids = outputs.get('out_obj_ids', [])
            masks   = outputs.get('out_binary_masks', [])   # (N, H, W) bool
            probs   = outputs.get('out_probs', [None] * len(obj_ids))

            for i, sam_id in enumerate(obj_ids):
                mask = masks[i] if i < len(masks) else None
                if mask is None or not mask.any():
                    continue
                H, W   = mask.shape
                rle    = mask_to_rle(mask)
                bbox   = bbox_from_mask(mask)
                area   = int(mask.sum())
                key    = int(sam_id)
                if key not in results:
                    results[key] = {}
                results[key][frame_name] = {
                    'mask_rle': rle,
                    'bbox_px':  bbox,
                    'area_px':  area,
                }

        return results

    finally:
        predictor.handle_request(dict(type='close_session', session_id=session_id))


def build_segments(predictor, image_dir, frame_names, nouns):
    """
    Run SAM3 for all nouns. Returns (objects_dict, missed_nouns).

    Output object IDs: "<noun_slug>_<sam3_obj_id>"
    e.g. "cup_3", "desk_0"
    """
    objects      = {}
    missed_nouns = []
    noun_to_type = {}
    for label in nouns.get('objects', []):
        noun_to_type[label] = 'object'
    for label in nouns.get('structures', []):
        noun_to_type[label] = 'structure'

    all_nouns = list(noun_to_type.keys())
    print(f'  {len(all_nouns)} nouns: {all_nouns}')

    for noun in all_nouns:
        print(f'  segmenting "{noun}"...')
        instances = run_noun(predictor, image_dir, frame_names, noun)

        if not instances:
            print(f'    WARN: no instances found for "{noun}"')
            missed_nouns.append(noun)
            continue

        slug = noun.replace(' ', '_')
        print(f'    {len(instances)} instance(s)')

        for sam_id, frame_data in instances.items():
            obj_id = f'{slug}_{sam_id}'
            if frame_data:
                objects[obj_id] = {
                    'label':  noun,
                    'type':   noun_to_type[noun],
                    'frames': frame_data,
                }

    return objects, missed_nouns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='scene directory (contains images/)')
    parser.add_argument('--nouns', required=True, help='nouns.json with objects/structures lists')
    parser.add_argument('--out',   required=True, help='output segments.json path')
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
    print(f'{len(frame_names)} frames  ({frame_names[0]} ... {frame_names[-1]})')
    print()

    print('loading SAM3...')
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()
    print()

    print('running segmentation...')
    objects, missed_nouns = build_segments(predictor, image_dir, frame_names, nouns)
    print()

    segments = {
        'objects':      objects,
        'nouns':        nouns,
        'missed_nouns': missed_nouns,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(segments, f)

    n_obj    = sum(1 for o in objects.values() if o['type'] == 'object')
    n_struct = sum(1 for o in objects.values() if o['type'] == 'structure')
    print(f'objects:    {n_obj}')
    print(f'structures: {n_struct}')
    print(f'missed:     {missed_nouns}')
    print(f'saved:      {out_path}')


if __name__ == '__main__':
    main()
