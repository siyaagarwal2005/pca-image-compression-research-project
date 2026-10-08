"""
src/plots.py

Visualization for the PCA compression experiment.

Two kinds of figures:

1. Rate-distortion curves
   PSNR (y) vs compression ratio (x). One curve per dataset, all on
   one plot so they can be compared at a glance. A second version
   uses explained variance on the y-axis as an orthogonal view.

2. Reconstruction examples
   For one dataset, show the same test image reconstructed at several
   k values side by side. This gives an intuitive, visual sense of
   what the PSNR numbers actually mean.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR


# ----------------------------------------------------------------------
# Rate-distortion plots
# ----------------------------------------------------------------------

def plot_rate_distortion(results_by_dataset, filename="rate_distortion.png"):
    """
    PSNR (y) vs compression ratio (x), one curve per dataset.

    Parameters
    ----------
    results_by_dataset : dict {dataset_name: list_of_result_dicts}
        Each result dict comes from pca_experiment.run_pca_sweep.
    filename           : output filename inside FIGURES_DIR
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for name, results in results_by_dataset.items():
        # x-axis: how much we compressed (bigger = more aggressive)
        x = [r["compression_ratio"] for r in results]
        # y-axis: how good the reconstruction is
        y = [r["psnr"] for r in results]

        plt.plot(x, y, marker="o", linewidth=1.8, label=name)

    # Reference line at the quality threshold — makes it easy to see
    # which datasets reach "acceptable quality" and at what cost.
    plt.axhline(
        y=30, color="gray", linestyle="--", linewidth=1,
        label="PSNR 30 dB (acceptability threshold)"
    )

    plt.xscale("log")   # compression ratio spans a wide range -> log scale
    plt.xlabel("Compression ratio (D / k), log scale — higher = more compressed")
    plt.ylabel("PSNR (dB) — higher = better reconstruction")
    plt.title("Rate–distortion curves across datasets")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_explained_variance(results_by_dataset, filename="explained_variance.png"):
    """
    Explained variance (y) vs k/D (x), one curve per dataset.

    Complementary view to PSNR: shows how much of the original
    information is retained at each level of compression.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for name, results in results_by_dataset.items():
        x = [r["k_over_D"] for r in results]
        y = [r["explained_variance"] for r in results]
        plt.plot(x, y, marker="o", linewidth=1.8, label=name)

    plt.axhline(
        y=0.95, color="gray", linestyle="--", linewidth=1,
        label="95% variance retained"
    )

    plt.xlabel("Fraction of original dimensions kept (k / D)")
    plt.ylabel("Explained variance (fraction)")
    plt.title("Information retained vs fraction of components used")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# Reconstruction examples
# ----------------------------------------------------------------------

def plot_reconstruction_examples(
    x_test_flat,
    img_shape,
    pca_components_fn,
    k_values,
    dataset_name,
    n_examples=4,
    filename=None,
):
    """
    Show test images reconstructed at several k values, side by side.

    Layout: rows = images, columns = original + each k value.

    Parameters
    ----------
    x_test_flat       : (N, D) test images, flattened & normalized
    img_shape         : (H, W) — to reshape vectors back to 2D for display
    pca_components_fn : callable(k) -> reconstructed (N, D) array
                        This lets the caller (run_experiment) pass in the
                        already-fitted PCA object without plots.py having
                        to import PCA itself.
    k_values          : iterable of k values to display
    dataset_name      : used in the title and default filename
    n_examples        : how many test images to show
    filename          : optional override for the output filename
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    H, W = img_shape
    k_values = list(k_values)
    n_cols = 1 + len(k_values)  # original + one column per k

    fig, axes = plt.subplots(
        n_examples, n_cols, figsize=(2.2 * n_cols, 2.2 * n_examples)
    )

    # Normalize axes to 2D array even when n_examples == 1
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    # Pick a few evenly-spaced test images
    indices = np.linspace(0, len(x_test_flat) - 1, n_examples, dtype=int)

    for row, idx in enumerate(indices):
        original = x_test_flat[idx].reshape(H, W)

        # Column 0: the original
        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        # Other columns: reconstructions at each k
        for col, k in enumerate(k_values, start=1):
            reconstructed = pca_components_fn(k)[idx].reshape(H, W)
            axes[row, col].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"k = {k}" if row == 0 else "")
            axes[row, col].axis("off")

    fig.suptitle(f"Reconstruction quality at different k — {dataset_name}")
    plt.tight_layout()

    if filename is None:
        safe = dataset_name.replace(" ", "_").replace("/", "_")
        filename = f"reconstructions_{safe}.png"

    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")