"""
src/metrics.py

Quality and cost metrics used to evaluate PCA-based image compression.

Each function takes a set of original images and their reconstructions
(both as 2D numpy arrays of shape (N, D), where D = H*W) and returns a
single number summarizing quality or cost.

Metrics implemented
-------------------
- MSE               : mean squared error between pixels
- PSNR              : MSE expressed in decibels (higher is better)
- SSIM              : structural similarity, closer to human perception
- explained_variance: fraction of original information retained
- compression_ratio : D / k, how many times smaller the data is

The functions are deliberately simple wrappers so that the entire project
uses one consistent definition of each metric.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import PIXEL_MAX


# ----------------------------------------------------------------------
# Error-based metrics
# ----------------------------------------------------------------------

def mse(original, reconstructed):
    """
    Mean squared error between two image sets.

    Parameters
    ----------
    original       : np.ndarray of shape (N, D), float in [0, 1]
    reconstructed  : np.ndarray of same shape

    Returns
    -------
    float — average squared difference across all pixels.
    Lower is better; 0 means perfect reconstruction.
    """
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original, reconstructed, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio in decibels.

    Intuition: how loud is the original signal compared to the error we
    introduced by reconstructing? Higher is better.

    Because we normalize pixels to [0, 1] in preprocessing, data_range
    defaults to 1.0. This is what makes PSNR comparable across datasets.

    Returns
    -------
    float — PSNR in dB. 30 dB is a common "acceptable quality" bar.
    """
    return float(
        peak_signal_noise_ratio(original, reconstructed, data_range=data_range)
    )


# ----------------------------------------------------------------------
# Structural similarity
# ----------------------------------------------------------------------

def ssim(original, reconstructed, img_shape, data_range=1.0):
    """
    Structural Similarity Index (SSIM).

    SSIM compares two images along three axes — luminance, contrast, and
    structure — and returns a score between -1 and 1 where 1 means the
    two images are identical. It captures edges and patterns, which makes
    it correlate with human perception better than PSNR.

    skimage.metrics.structural_similarity expects 2D or 3D arrays, so
    we reshape (N, D) back into (N, H, W) to compute SSIM per image,
    then average across the set.

    Parameters
    ----------
    original, reconstructed : (N, D) arrays
    img_shape               : (H, W) — needed to reshape back to images

    Returns
    -------
    float — average SSIM across all images, in [0, 1] for natural images.
    """
    N = original.shape[0]
    orig_2d = original.reshape(N, *img_shape)
    rec_2d = reconstructed.reshape(N, *img_shape)

    scores = [
        structural_similarity(
            orig_2d[i], rec_2d[i], data_range=data_range
        )
        for i in range(N)
    ]
    return float(np.mean(scores))


# ----------------------------------------------------------------------
# Cost metrics
# ----------------------------------------------------------------------

def compression_ratio(k, D):
    """
    How many times smaller the compressed data is than the original.

    D = original dimensionality (H*W)
    k = number of PCA components kept

    A ratio of 16 means the compressed version is 16× smaller than the
    original pixels. Bigger compression = more aggressive, usually
    worse quality. This is what "compression ratio" normally means.
    """
    return D / k


def fraction_kept(k, D):
    """
    Fraction of original dimensions kept (the reciprocal view).

    k/D is what you get if you want to say "I kept X% of the original
    data" rather than "the data is X times smaller." Both are useful;
    using both makes it easy to compare across datasets of different D.
    """
    return k / D