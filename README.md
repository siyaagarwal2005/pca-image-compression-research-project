# A Cross-Dataset Study of Optimal PCA Dimensionality for Image Compression

**Research Question:** How does image type affect the number of PCA
components required to achieve a good trade-off between image quality
and compression?

## Datasets

| Dataset        | Content              | Image size | Dimensionality |
|----------------|----------------------|------------|----------------|
| MNIST          | Handwritten digits   | 28 × 28    | 784            |
| Fashion-MNIST  | Clothing items       | 28 × 28    | 784            |
| Olivetti Faces | Human faces          | 64 × 64    | 4096           |
| CIFAR-10       | Natural objects      | 32 × 32    | 1024           |

CIFAR-10 is converted to grayscale for the main experiment.

## Project Structure

    .
    ├── config.py              # All experimental constants
    ├── requirements.txt
    ├── run_load_check.py      # Verifies datasets load correctly
    ├── src/
    │   ├── load_data.py       # Loads all four datasets
    │   └── preprocess.py      # Flatten + normalize
    └── results/
        ├── figures/
        └── tables/

## Setup

    pip install -r requirements.txt

## Verify Datasets

    python run_load_check.py

Datasets are downloaded automatically on first run by TensorFlow and
scikit-learn; nothing is stored in the repository.

## Status

- [x] Project scaffolding
- [x] Dataset loading and preprocessing
- [x] PCA experiment (fit, sweep k, reconstruct)
- [x] Metrics (MSE, PSNR, SSIM, explained variance)
- [x] Optimal-k analysis (threshold + elbow)
- [x] Rate–distortion plots and final report