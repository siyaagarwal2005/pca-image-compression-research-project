"""
src/analysis.py

Turn raw metric results (from pca_experiment.run_pca_sweep) into
"optimal k" answers under two definitions:

1. Threshold-based
   The smallest k for which PSNR reaches the acceptability threshold
   (PSNR_THRESHOLD_DB, default 30 dB). This answers:
       "How many components do I need to reach acceptable quality?"

2. Elbow-based
   The first k at which adding more components stops yielding a
   meaningful improvement (improvement drops below
   ELBOW_IMPROVEMENT_DB, default 0.5 dB per step). This answers:
       "At what point do additional components stop being worth it?"

Reporting both is important because they can disagree — one is a
minimum requirement, the other is a practical stopping point.
"""

from config import PSNR_THRESHOLD_DB, ELBOW_IMPROVEMENT_DB


# ----------------------------------------------------------------------
# Threshold-based optimal k
# ----------------------------------------------------------------------

def optimal_k_threshold(results, threshold=PSNR_THRESHOLD_DB):
    """
    Smallest k such that PSNR >= threshold.

    Parameters
    ----------
    results   : list of dicts from run_pca_sweep, sorted by increasing k
    threshold : PSNR threshold in dB (default from config)

    Returns
    -------
    int or None — the smallest qualifying k, or None if no k reaches
    the threshold (which is itself a finding: the dataset compresses
    poorly under PCA at any tested k).
    """
    for r in results:
        if r["psnr"] >= threshold:
            return r["k"]
    return None


# ----------------------------------------------------------------------
# Elbow-based optimal k
# ----------------------------------------------------------------------

def optimal_k_elbow(results, min_improvement=ELBOW_IMPROVEMENT_DB):
    """
    First k where the step-to-step PSNR improvement drops below
    `min_improvement` dB.

    Intuition
    ---------
    The PSNR-vs-k curve is steep at first and flattens out. The elbow is
    where it transitions — the point beyond which adding components
    earns diminishing returns.

    Method
    ------
    - Compute PSNR deltas between consecutive k values.
    - Find the first delta that drops below `min_improvement`.
    - Return the k value at the START of that step (the last k that
      still gave a meaningful improvement).

    Edge cases
    ----------
    - If no step ever drops below the threshold, return the largest k
      (the curve never really flattened within the tested range).
    - If the very first step already drops below the threshold, return
      the first k (the curve is flat from the start).
    """
    if len(results) < 2:
        # Not enough points to compute a slope — nothing to find.
        return results[0]["k"] if results else None

    psnrs = [r["psnr"] for r in results]
    ks    = [r["k"]    for r in results]

    # Step-to-step improvement
    improvements = [psnrs[i + 1] - psnrs[i] for i in range(len(psnrs) - 1)]

    # Find first index where improvement is below threshold
    for i, imp in enumerate(improvements):
        if imp < min_improvement:
            # ks[i] is the last k that still improved meaningfully.
            return ks[i]

    # Curve never flattened within the tested range — return largest k
    return ks[-1]


# ----------------------------------------------------------------------
# Convenience: compute everything for one dataset
# ----------------------------------------------------------------------

def summarize_results(results):
    """
    Given results for one dataset, return a dict with:

        k_threshold, k_elbow,
        best_psnr, best_ssim, best_k,
        psnr_at_threshold (informational)

    This is what goes into the final summary table.
    """
    k_thresh = optimal_k_threshold(results)
    k_elbow  = optimal_k_elbow(results)

    best = max(results, key=lambda r: r["psnr"])

    return {
        "k_threshold": k_thresh,
        "k_elbow":     k_elbow,
        "best_k":      best["k"],
        "best_psnr":   best["psnr"],
        "best_ssim":   best["ssim"],
    }