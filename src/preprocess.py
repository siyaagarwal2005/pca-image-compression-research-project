"""
src/preprocess.py

Preprocessing utilities: convert image stacks into flat vectors and
scale pixel values to [0, 1].

PCA in scikit-learn expects a 2D array of shape (n_samples, n_features),
so we reshape every (N, H, W) stack into (N, H*W).
"""

import numpy as np

from config import PIXEL_MAX


def flatten(images):
    """
    Reshape (N, H, W) -> (N, H*W).

    Example: MNIST 28x28 -> 784-dimensional vectors
             Olivetti 64x64 -> 4096-dimensional vectors
    """
    n = images.shape[0]
    return images.reshape(n, -1)


def normalize(images):
    """
    Scale pixel intensities to [0, 1].

    Datasets come in as uint8 (0-255) or float32 (already 0-1).
    Dividing by 255 on an already-normalized array would be wrong,
    so we check the dtype.
    """
    if images.dtype == np.uint8:
        return images.astype(np.float32) / PIXEL_MAX
    # Olivetti is already float32 in [0, 1]
    return images.astype(np.float32)


def prepare(images):
    """
    Full preprocessing pipeline: normalize then flatten.

    Order matters — normalizing after flattening would also work, but
    doing it before makes the intent clearer.
    """
    return flatten(normalize(images))