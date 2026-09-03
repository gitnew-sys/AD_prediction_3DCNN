import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize a 3D saliency .npy volume")
    parser.add_argument("npy_path", help="Path to ad_nc_saliency_map.npy")
    parser.add_argument("--output", default="ad_nc_saliency_slices.png")
    args = parser.parse_args()

    saliency = np.load(args.npy_path).astype(np.float32)
    if saliency.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape={saliency.shape}")

    # Use robust limits so a few extreme voxels do not dominate the color map.
    vmax = float(np.percentile(saliency, 99.5))
    if vmax <= 0:
        vmax = float(saliency.max())
    if vmax <= 0:
        vmax = 1.0

    d, h, w = saliency.shape
    centers = (d // 2, h // 2, w // 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        (saliency[centers[0], :, :], f"Axial: D={centers[0]}"),
        (saliency[:, centers[1], :], f"Coronal: H={centers[1]}"),
        (saliency[:, :, centers[2]], f"Sagittal: W={centers[2]}"),
    ]

    for ax, (image, title) in zip(axes, views):
        ax.imshow(np.rot90(image), cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"AD/NC saliency map | shape={saliency.shape} | "
        f"min={saliency.min():.4g}, max={saliency.max():.4g}"
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="hot"),
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="saliency intensity",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
