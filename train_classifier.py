"""
Evaluate the k-NN relevance scorer against current hand labels.

Runs leave-one-out validation: each hand-labeled RFP in output/rfps_all.csv
is scored with its own entry excluded from its neighbour pool (matcher/score.py
does this automatically via opportunity_id), then compared against the label
it was actually given. Reports accuracy and ROC-AUC.

This does not train or persist a model — matcher/score.py always scores live
from whatever is currently labeled in rfps_all.csv. Use this script just to
sanity-check scoring quality after adding labels.

Usage:
    python train_classifier.py
"""

import sys

import numpy as np
import pandas as pd

from matcher.score import LGeoMatcher, LABEL_COL, REAL_LABELS_PATH

MIN_POSITIVES = 5


def _roc_auc(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Rank-based ROC-AUC (probability a random positive outscores a random negative)."""
    order = np.argsort(predicted)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(predicted) + 1)
    n_pos = int((actual == 1).sum())
    n_neg = int((actual == 0).sum())
    sum_ranks_pos = ranks[actual == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    try:
        df = pd.read_csv(REAL_LABELS_PATH, encoding="latin-1")
    except FileNotFoundError:
        print(f"{REAL_LABELS_PATH} not found. Run run.py and combine_output.py first.")
        sys.exit(1)

    if LABEL_COL not in df.columns:
        print(
            f"No '{LABEL_COL}' column in {REAL_LABELS_PATH}.\n"
            f"Add a '{LABEL_COL}' column (1=relevant, 0=not relevant) to some rows, then re-run."
        )
        sys.exit(1)

    labeled = df[pd.to_numeric(df[LABEL_COL], errors="coerce").notna()].copy()
    labeled[LABEL_COL] = pd.to_numeric(labeled[LABEL_COL]).astype(int)
    if labeled.empty:
        print(f"No labeled rows found. Add values to the '{LABEL_COL}' column first.")
        sys.exit(1)

    n_pos = int((labeled[LABEL_COL] == 1).sum())
    n_neg = int((labeled[LABEL_COL] == 0).sum())
    print(f"Labeled examples: {len(labeled)} total ({n_pos} relevant, {n_neg} not relevant)")
    if n_pos < MIN_POSITIVES:
        print(f"  Note: fewer than {MIN_POSITIVES} positive examples — evaluation will be noisy.")

    print("\nRunning leave-one-out evaluation ...")
    matcher = LGeoMatcher()
    rows = labeled.to_dict("records")
    predicted = np.array([matcher.score(r)["relevance_score"] for r in rows])
    actual = labeled[LABEL_COL].to_numpy()

    predicted_label = (predicted >= 0.5).astype(int)
    accuracy = float((predicted_label == actual).mean())
    print(f"Leave-one-out accuracy @ 0.5 threshold: {accuracy:.3f}")

    if n_pos >= 2 and n_neg >= 2:
        auc = _roc_auc(actual, predicted)
        print(f"Leave-one-out ROC-AUC: {auc:.3f}")
        if auc < 0.75:
            print("  Note: ROC-AUC below 0.75 — consider labeling more data.")
    else:
        print("  Not enough of both classes for ROC-AUC (need >= 2 positive and >= 2 negative).")


if __name__ == "__main__":
    main()
