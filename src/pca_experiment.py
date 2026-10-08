"""
src/pca_experiment.py

Core experiment: fit PCA on the training set, sweep different k values,
reconstruct the test set at each k, and return metrics for every k.

Design notes
------------
1. PCA is fit ONLY on the training set. Fitting on test data would leak
   information and make the reported quality dishonest.

2. PCA is fit ONCE with the largest k we will need. Because the principal
   components are nested (the first component of PCA(k=100) is identical
   to the first component of PCA(k=5)), we can slice off the first k
   components for any smaller k. This is both faster and yields a
   consistent basis across all k values.

3. Reconstruction is done manually: 
       X ≈ Z @ components[:k] + mean
   where Z = X_centered @ components[:k].T. This lets us control exactly
   how many components are used, without refitting PCA each time.
"""

import numpy as np
from sklearn.decomposition import PCA

from src.metrics import mse, psnr, ssim, compression_ratio, fraction_kept
from config import RANDOM_SEED


def fit_pca(x_train_flat, max_k):
    """
    Fit PCA on the training set with `max_k` components.

    Parameters
    ----------
    x_train_flat : np.ndarray of shape (N_train, D)
        Flattened, normalized training images.
    max_k : int
        Maximum number of components to keep. We cap this at
        min(N_train, D) because PCA cannot produce more components
        than there are samples or features, whichever is smaller.

    Returns
    -------
    sklearn.decomposition.PCA
        A fitted PCA object whose `.components_` has shape (max_k, D)
        and whose `.mean_` is the training mean.
    """
    # PCA can't produce more components than min(N, D). We also guard
    # against silly max_k values by clamping.
    max_k = min(max_k, x_train_flat.shape[0], x_train_flat.shape[1])

    pca = PCA(n_components=max_k, random_state=RANDOM_SEED)
    pca.fit(x_train_flat)
    return pca


def reconstruct(pca, x_flat, k):
    """
    Reconstruct images using only the first k principal components.

    Procedure
    ---------
    1. Center the data:   X_c = X - mean
    2. Project:           Z   = X_c @ components[:k].T   -> (N, k)
    3. Reconstruct:       X_r = Z @ components[:k] + mean

    Parameters
    ----------
    pca    : fitted sklearn PCA object
    x_flat : (N, D) array of flattened images to reconstruct
    k      : int, number of components to use

    Returns
    -------
    np.ndarray of shape (N, D) — the reconstructed images.
    """
    # Step 1: center using the training mean (pca.mean_)
    x_centered = x_flat - pca.mean_

    # Step 2: project down to k components
    components_k = pca.components_[:k]              # shape (k, D)
    z = x_centered @ components_k.T                 # shape (N, k)

    # Step 3: reconstruct back to D dimensions
    x_reconstructed = z @ components_k + pca.mean_  # shape (N, D)

    return x_reconstructed


def run_pca_sweep(x_train_flat, x_test_flat, img_shape, k_values):
    """
    Run the full PCA compression experiment on one dataset.

    For each k in k_values:
      - reconstruct the test set using only k components
      - compute MSE, PSNR, SSIM, explained variance, compression ratio

    Parameters
    ----------
    x_train_flat : (N_train, D) training images, flattened & normalized
    x_test_flat  : (N_test,  D) test images, flattened & normalized
    img_shape    : (H, W) — needed by SSIM to reshape vectors to images
    k_values     : iterable of ints — which k values to test

    Returns
    -------
    list of dicts — one per k value, each containing:
        k, k_over_D, compression_ratio,
        explained_variance, mse, psnr, ssim
    """
    D = x_train_flat.shape[1]
    max_k = max(k_values)

    # Fit PCA once with the largest k we'll need.
    # Components are nested, so we can slice smaller k values later.
    pca = fit_pca(x_train_flat, max_k)

    results = []

    for k in k_values:
        # Skip k values that exceed what the fitted PCA can provide.
        # (Happens if a dataset has fewer samples than the largest k.)
        if k > pca.components_.shape[0]:
            continue

        # Compress + decompress the test set using only k components
        x_rec = reconstruct(pca, x_test_flat, k)

        # Explained variance = sum of variance ratios of first k comps
        explained = float(np.sum(pca.explained_variance_ratio_[:k]))

        results.append({
            "k":                  k,
            "k_over_D":           fraction_kept(k, D),
            "compression_ratio":  compression_ratio(k, D),
            "explained_variance": explained,
            "mse":                mse(x_test_flat, x_rec),
            "psnr":               psnr(x_test_flat, x_rec),
            "ssim":               ssim(x_test_flat, x_rec, img_shape),
        })

    return results