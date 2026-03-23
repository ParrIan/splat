"""
Visualize DINOv2 patch features on a single image using PCA.
If DINOv2 is working, similar regions (desk surface, keyboard, monitor)
should have similar colors in the output.

Usage:
    python pipeline/dino_sanity.py --image path/to/frame.jpg
"""

import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DINO_PATCH = 14

dino_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()

    img = Image.open(args.image).convert('RGB')
    W, H = img.size

    # resize to nearest multiple of 14
    new_w = (W // DINO_PATCH) * DINO_PATCH
    new_h = (H // DINO_PATCH) * DINO_PATCH
    img_resized = img.resize((new_w, new_h))

    x = dino_transform(img_resized).unsqueeze(0)

    with torch.no_grad():
        feats = model.get_intermediate_layers(x, n=1)[0]  # (1, num_patches, 768)

    grid_h = new_h // DINO_PATCH
    grid_w = new_w // DINO_PATCH
    feats = feats[0].numpy()  # (num_patches, 768)

    # PCA to 3 dims -> RGB visualization
    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(feats)  # (num_patches, 3)

    # normalize to [0, 1]
    feats_pca -= feats_pca.min(axis=0)
    feats_pca /= feats_pca.max(axis=0) + 1e-6

    feat_img = feats_pca.reshape(grid_h, grid_w, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_resized)
    axes[0].set_title('original')
    axes[0].axis('off')

    axes[1].imshow(feat_img)
    axes[1].set_title('DINOv2 patch features (PCA RGB)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('dino_sanity.png', dpi=150)
    print('saved dino_sanity.png')
    plt.show()


if __name__ == "__main__":
    main()
