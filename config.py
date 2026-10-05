"""
config.py

Central configuration for the entire project. Every constant, threshold,
and experimental parameter lives here so that changing a value in one
place updates the whole pipeline.

"""

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
# Used anywhere randomness is involved (train/test split, PCA init, etc.)
# so that every run gives the same result.
RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
# Names used as keys throughout the project. Order matters for plots.
DATASET_NAMES = ["MNIST", "Fashion-MNIST", "Olivetti", "CIFAR-10"]


# ----------------------------------------------------------------------
# PCA sweep — values of k (number of components) to test
# ----------------------------------------------------------------------
# We start small and grow. Each dataset may not use all of these; the
# experiment loop skips k values larger than the dataset's dimensionality.
K_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]


# ----------------------------------------------------------------------
# Quality thresholds — used later to define "good enough"
# ----------------------------------------------------------------------
PSNR_THRESHOLD_DB = 30.0       # Widely used in image-compression literature
SSIM_THRESHOLD = 0.90          # "Structurally almost identical"

# Elbow detection: if PSNR improves by less than this between two
# consecutive k values, we consider it "diminishing returns".
ELBOW_IMPROVEMENT_DB = 0.5


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
# Pixel values are scaled from [0, 255] to [0, 1] before PCA.
# This keeps MSE / PSNR comparable across datasets.
PIXEL_MAX = 255.0