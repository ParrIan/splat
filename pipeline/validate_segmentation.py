"""
Validate SAM3 segmentation output.

Checks:
  1. Every noun phrase has at least one instance found
  2. Object IDs are consistent across frames (same object = same ID)
  3. Mask IoU between adjacent frames for same object ID > threshold
  4. Visualizes masks color-coded by object ID on sampled frames

Usage:
    python pipeline/validate_segmentation.py \
        --scene  "splat_session 2" \
        --segments pipeline/segments.json \
        --frames 6

Input (segments.json schema):
    {
      "objects": {
        "<object_id>": {
          "label": str,
          "type": "object" | "structure",
          "frames": {
            "<frame_name>": {
              "mask_rle": {"counts": str, "size": [H, W]},
              "bbox_px": [x0, y0, x1, y1],
              "area_px": int
            }
          }
        }
      },
      "nouns": {
        "objects": [str, ...],
        "structures": [str, ...]
      },
      "missed_nouns": [str, ...]
    }
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

IOU_THRESHOLD = 0.5


def rle_decode(rle, shape):
    """Decode COCO RLE mask to binary numpy array."""
    counts = rle['counts']
    if isinstance(counts, str):
        # compressed RLE string from pycocotools
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(bool)
    # uncompressed counts list
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos+c] = True
        pos += c
    return mask.reshape(shape)


def compute_iou(mask_a, mask_b):
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return intersection / union if union > 0 else 0.0


def check_missed_nouns(segments):
    missed = segments.get('missed_nouns', [])
    if missed:
        print(f'WARN  missed nouns (no instances found): {missed}')
    else:
        print('PASS  all noun phrases have at least one instance')
    return len(missed) == 0


def check_id_consistency(segments, iou_threshold):
    objects = segments['objects']
    failures = []

    for obj_id, obj in objects.items():
        frames = obj['frames']
        frame_names = sorted(frames.keys())
        if len(frame_names) < 2:
            continue

        for i in range(len(frame_names) - 1):
            fn_a = frame_names[i]
            fn_b = frame_names[i + 1]
            rle_a = frames[fn_a]['mask_rle']
            rle_b = frames[fn_b]['mask_rle']
            size  = rle_a['size']

            mask_a = rle_decode(rle_a, size)
            mask_b = rle_decode(rle_b, size)
            iou    = compute_iou(mask_a, mask_b)

            if iou < iou_threshold:
                failures.append((obj_id, fn_a, fn_b, iou))

    if failures:
        print(f'FAIL  {len(failures)} adjacent-frame ID consistency failures (IoU < {iou_threshold}):')
        for obj_id, fn_a, fn_b, iou in failures[:5]:
            print(f'      {obj_id}: {fn_a} -> {fn_b}  IoU={iou:.3f}')
        if len(failures) > 5:
            print(f'      ... and {len(failures)-5} more')
    else:
        print(f'PASS  all adjacent-frame object IDs consistent (IoU >= {iou_threshold})')

    return len(failures) == 0


def visualize(segments, image_dir, n_frames):
    objects  = segments['objects']
    obj_ids  = list(objects.keys())
    colors   = plt.cm.tab20(np.linspace(0, 1, max(len(obj_ids), 1)))
    id_color = {oid: colors[i] for i, oid in enumerate(obj_ids)}

    # collect all frame names, sample evenly
    all_frames = set()
    for obj in objects.values():
        all_frames.update(obj['frames'].keys())
    all_frames = sorted(all_frames)
    step = max(1, len(all_frames) // n_frames)
    sampled = all_frames[::step][:n_frames]

    fig, axes = plt.subplots(1, len(sampled), figsize=(5 * len(sampled), 5))
    if len(sampled) == 1:
        axes = [axes]

    for ax, frame_name in zip(axes, sampled):
        img_path = image_dir / frame_name
        if not img_path.exists():
            ax.set_title(f'missing: {frame_name}')
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        H, W = img.shape[:2]
        overlay = img.copy().astype(float)

        for obj_id, obj in objects.items():
            if frame_name not in obj['frames']:
                continue
            rle  = obj['frames'][frame_name]['mask_rle']
            mask = rle_decode(rle, rle['size'])
            if mask.shape != (H, W):
                continue
            color = (np.array(id_color[obj_id][:3]) * 255).astype(float)
            overlay[mask] = overlay[mask] * 0.4 + color * 0.6

        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(frame_name, fontsize=8)
        ax.axis('off')

    patches = [mpatches.Patch(color=id_color[oid], label=f'{oid} ({objects[oid]["label"]})')
               for oid in obj_ids]
    fig.legend(handles=patches, loc='lower center', ncol=min(6, len(obj_ids)), fontsize=7)
    plt.tight_layout()
    out = Path('pipeline/validate_segmentation.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'\nvisualization saved: {out}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',    required=True)
    parser.add_argument('--segments', required=True)
    parser.add_argument('--frames',   type=int, default=6)
    parser.add_argument('--iou',      type=float, default=IOU_THRESHOLD)
    args = parser.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)

    image_dir = Path(args.scene) / 'images'
    n_objects = len(segments['objects'])
    print(f'objects found: {n_objects}')
    print(f'noun phrases:  {segments["nouns"]}')
    print()

    passed = True
    passed &= check_missed_nouns(segments)
    passed &= check_id_consistency(segments, args.iou)

    print()
    print('PASS' if passed else 'FAIL')

    visualize(segments, image_dir, args.frames)


if __name__ == '__main__':
    main()
