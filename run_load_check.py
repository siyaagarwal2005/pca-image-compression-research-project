"""
run_load_check.py

Sanity-check script: loads all four datasets and prints their shapes
and value ranges. Use this to verify that the environment is set up
correctly and that every dataset is available.

Run:
    python run_load_check.py
"""

from src.load_data import load_all
from src.preprocess import prepare


def main():
    print("Loading datasets... (first run may take a minute)\n")

    datasets = load_all()

    # Header
    print(f"{'Dataset':<15} {'Train shape':<22} {'Test shape':<22} {'Flattened D':<12}")
    print("-" * 75)

    for name, (x_train, x_test) in datasets.items():
        # Preprocess training data to get the flattened dimensionality
        train_flat = prepare(x_train)

        print(
            f"{name:<15} "
            f"{str(x_train.shape):<22} "
            f"{str(x_test.shape):<22} "
            f"{train_flat.shape[1]:<12}"
        )

    print("\nAll datasets loaded successfully.")

    # Quick sanity check on value ranges for one dataset
    print("\nSanity check — pixel value ranges (before / after normalization):")
    for name, (x_train, _) in datasets.items():
        raw_min, raw_max = x_train.min(), x_train.max()
        norm = prepare(x_train)
        norm_min, norm_max = norm.min(), norm.max()
        print(
            f"  {name:<15} raw [{raw_min:.0f}, {raw_max:.0f}]  "
            f"normalized [{norm_min:.2f}, {norm_max:.2f}]"
        )


if __name__ == "__main__":
    main()