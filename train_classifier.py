"""
Evaluate the k-NN relevance scorer against current hand labels, across
every site (output/*/rfps_all.csv).

Runs leave-one-out validation: each hand-labeled RFP is scored with its own
entry excluded from its neighbour pool (matcher/score.py does this
automatically via opportunity_id), then compared against the label it was
actually given. Reports accuracy and ROC-AUC.

This does not train or persist a model — matcher/score.py always scores live
from whatever is currently labeled. Use this script just to sanity-check
scoring quality after adding labels.

Usage:
    python train_classifier.py
"""

import sys

import numpy as np

from matcher.score import LGeoMatcher, load_labeled_dataframe

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
    df = load_labeled_dataframe()
    if df.empty:
        print("No labeled rows found across output/*/rfps_all.csv. Run run.py and combine_output.py first, then add labels.")
        sys.exit(1)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"Labeled examples: {len(df)} total ({n_pos} relevant, {n_neg} not relevant)")
    if n_pos < MIN_POSITIVES:
        print(f"  Note: fewer than {MIN_POSITIVES} positive examples — evaluation will be noisy.")

    print("\nRunning leave-one-out evaluation ...")
    matcher = LGeoMatcher()
    rows = df.to_dict("records")
    predicted = np.array([matcher.score(r)["relevance_score"] for r in rows])
    actual = df["label"].to_numpy()

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
