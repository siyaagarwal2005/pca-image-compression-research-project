"""
run_experiment.py

Entry point for the cross-dataset PCA compression study.

For each of the four datasets, this script:
  1. Loads and preprocesses the data
  2. Fits PCA on the training set and sweeps k values
  3. Computes optimal-k under threshold and elbow definitions
  4. Produces rate–distortion and reconstruction figures
  5. Saves per-dataset results and a global summary table

Run from the project root:

    python run_experiment.py

Outputs land in results/tables/ and results/figures/.
"""

import os
import pandas as pd

from config import (
    K_VALUES,
    RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    PSNR_THRESHOLD_DB,
)
from src.load_data import load_all
from src.preprocess import prepare
from src.pca_experiment import fit_pca, reconstruct, run_pca_sweep
from src.analysis import optimal_k_threshold, optimal_k_elbow, summarize_results
from src.plots import (
    plot_rate_distortion,
    plot_explained_variance,
    plot_reconstruction_examples,
)


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def process_dataset(name, x_train, x_test):
    """
    Run the full pipeline for one dataset.

    Returns
    -------
    results   : list of dicts (one per k) from the PCA sweep
    summary   : dict with optimal-k and best-quality info
    pca       : fitted PCA object (needed later for reconstruction plots)
    img_shape : (H, W) tuple
    x_test_flat : preprocessed test set (needed later for plots)
    """
    img_shape = x_train.shape[1:]   # (H, W)

    print(f"\n=== {name} ===")
    print(f"  Image shape:   {img_shape}")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Test samples:  {x_test.shape[0]}")

    # 1. Preprocess (flatten + normalize)
    x_train_flat = prepare(x_train)
    x_test_flat  = prepare(x_test)
    D = x_train_flat.shape[1]
    print(f"  Flattened D:   {D}")

    # 2. Filter k values to what's feasible for this dataset.
    #    PCA cannot return more components than min(N_train, D),
    #    so we drop k values above that bound.
    max_k_feasible = min(x_train_flat.shape[0], D)
    k_values = [k for k in K_VALUES if k <= max_k_feasible]
    print(f"  Testing k:     {k_values}")

    # 3. Run the sweep
    print("  Running PCA sweep...")
    results = run_pca_sweep(x_train_flat, x_test_flat, img_shape, k_values)

    # 4. Summarize: optimal k under both definitions
    summary = summarize_results(results)
    summary["dataset"] = name
    summary["D"] = D

    # Also store the threshold-used and the k/D fraction of the
    # threshold-optimal k, for the final comparison table.
    summary["psnr_threshold_used"] = PSNR_THRESHOLD_DB
    if summary["k_threshold"] is not None:
        summary["k_threshold_over_D"] = summary["k_threshold"] / D
        summary["k_threshold_compression_ratio"] = D / summary["k_threshold"]
    else:
        summary["k_threshold_over_D"] = None
        summary["k_threshold_compression_ratio"] = None

    # Refit PCA (or reuse) — we need the object for reconstruction plots.
    # run_pca_sweep already fitted one internally; refitting here is a
    # small extra cost for a clean interface. On these dataset sizes it's
    # still fast. If it ever becomes slow, expose the PCA from the sweep.
    pca = fit_pca(x_train_flat, max(k_values))

    # Print a compact per-dataset result table
    df = pd.DataFrame(results)
    print("  Results:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return results, summary, pca, img_shape, x_test_flat


def main():
    ensure_dirs()

    print("Loading all datasets...")
    datasets = load_all()

    all_results = {}
    summaries   = []

    # ---------------------------------------------------------------
    # Main loop: run the pipeline for each dataset
    # ---------------------------------------------------------------
    for name, (x_train, x_test) in datasets.items():
        results, summary, pca, img_shape, x_test_flat = process_dataset(
            name, x_train, x_test
        )

        # Save full per-k results for this dataset
        df = pd.DataFrame(results)
        out_csv = os.path.join(TABLES_DIR, f"full_results_{name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        # Collect for summary and plotting
        all_results[name] = results
        summaries.append(summary)

        # -----------------------------------------------------------
        # Reconstruction examples for this dataset.
        # We pass a small closure so plots.py doesn't need PCA.
        # Pick a few k values spread across the range.
        # -----------------------------------------------------------
        recon_ks = [k for k in [5, 20, 50, 100, 200] if k <= max(r["k"] for r in results)]
        if recon_ks:
            plot_reconstruction_examples(
                x_test_flat=x_test_flat,
                img_shape=img_shape,
                pca_components_fn=lambda k, pca=pca, xf=x_test_flat:
                    reconstruct(pca, xf, k),
                k_values=recon_ks,
                dataset_name=name,
            )

    # ---------------------------------------------------------------
    # Global summary table
    # ---------------------------------------------------------------
    summary_df = pd.DataFrame(summaries)[[
        "dataset", "D",
        "k_threshold", "k_threshold_over_D", "k_threshold_compression_ratio",
        "k_elbow",
        "best_k", "best_psnr", "best_ssim",
    ]]
    summary_path = os.path.join(TABLES_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    print("\n=== Summary of optimal-k across datasets ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------
    # Global figures
    # ---------------------------------------------------------------
    print("\nGenerating figures...")
    plot_rate_distortion(all_results)
    plot_explained_variance(all_results)

    print("\nDone. Outputs in results/figures/ and results/tables/.")


if __name__ == "__main__":
    main()