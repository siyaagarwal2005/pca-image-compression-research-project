"""
src/load_data.py

Loads all four datasets used in this project.

Every loader returns a tuple (x_train, x_test) where both are numpy
arrays of shape (N, H, W) — that is, a stack of grayscale 2D images.

Notes
-----
- MNIST, Fashion-MNIST and CIFAR-10 come pre-split into train/test and
are downloaded automatically by TensorFlow on first use.
- Olivetti Faces comes from scikit-learn and is NOT pre-split, so we
split it ourselves using a fixed random seed for reproducibility.
- CIFAR-10 is RGB (32, 32, 3); we convert to grayscale for the main
experiment as stated in the project proposal.
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED


# ----------------------------------------------------------------------
# Individual loaders
# ----------------------------------------------------------------------

def load_mnist():
    """Handwritten digits. Returns (train, test) as (N, 28, 28) uint8."""
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    return x_train, x_test


def load_fashion_mnist():
    """Clothing items. Returns (train, test) as (N, 28, 28) uint8."""
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    return x_train, x_test


def load_olivetti():
    """
    Human faces (400 images of 40 people, 64x64 grayscale).

    Olivetti has no official train/test split, so we make one ourselves:
    75% train, 25% test, using RANDOM_SEED for reproducibility.
    """
    faces = fetch_olivetti_faces()
    images = faces.images  # shape (400, 64, 64), float32 in [0, 1]

    x_train, x_test = train_test_split(
        images,
        test_size=0.25,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return x_train, x_test


def load_cifar10_grayscale():
    """
    Natural objects/animals (RGB 32x32). Converted to grayscale.

    tf.image.rgb_to_grayscale returns shape (N, 32, 32, 1); we squeeze
    the last dimension to get (N, 32, 32), consistent with other loaders.
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    x_train = tf.image.rgb_to_grayscale(x_train).numpy().squeeze(-1)
    x_test = tf.image.rgb_to_grayscale(x_test).numpy().squeeze(-1)

    return x_train, x_test


# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------

def load_all():
    """
    Load every dataset and return a dict:

        {
            "MNIST":         (x_train, x_test),
            "Fashion-MNIST": (x_train, x_test),
            "Olivetti":      (x_train, x_test),
            "CIFAR-10":      (x_train, x_test),
        }

    Each x_* is a numpy array of shape (N, H, W).
    """
    return {
        "MNIST":         load_mnist(),
        "Fashion-MNIST": load_fashion_mnist(),
        "Olivetti":      load_olivetti(),
        "CIFAR-10":      load_cifar10_grayscale(),
    }