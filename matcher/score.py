"""
Relevance scoring for RFP records against LGeo's service profile.

Every RFP is scored by weighted k-nearest-neighbour similarity, in
embedding space, to a single pool of labeled example texts: the curated
seed examples in matcher/profiles.py plus whatever the user has
hand-labeled in output/rfps_all.csv (the "my assessment" column). There
is no persisted model and no separate training step — labeling a row in
rfps_all.csv changes scores the next time this runs. See
train_classifier.py for an (optional) evaluation of scoring quality
against current labels.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from matcher.profiles import SEED_EXAMPLES

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.5  # "majority of similar examples are positive"
K_NEIGHBORS = 10
SEED_WEIGHT = 1.0
REAL_WEIGHT = 2.0  # hand labels outweigh seed examples as they accumulate
LABEL_COL = "my assessment"
REAL_LABELS_PATH = "output/bcbids/rfps_all.csv"
NEAREST_EXAMPLE_MAXLEN = 150


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(filter(None, [
        str(row.get("opportunity_description") or ""),
        str(row.get("commodities") or ""),
        str(row.get("type") or ""),
        str(row.get("summary") or ""),
    ]))


class LGeoMatcher:
    """Load once, score many."""

    @cached_property
    def _model(self) -> SentenceTransformer:
        print(f"  Loading embedding model ({MODEL_NAME}) ...")
        return SentenceTransformer(MODEL_NAME)

    @cached_property
    def _seed_embeddings(self) -> np.ndarray:
        texts = [t for t, _ in SEED_EXAMPLES]
        return self._model.encode(texts, normalize_embeddings=True)

    def _load_real_examples(self) -> list[tuple[str, int, str | None]]:
        """Hand-labeled rows from rfps_all.csv as (text, label, opportunity_id)."""
        if not os.path.exists(REAL_LABELS_PATH):
            return []
        try:
            df = pd.read_csv(REAL_LABELS_PATH, encoding="latin-1")
        except Exception:
            return []
        if LABEL_COL not in df.columns:
            return []
        labeled = df[pd.to_numeric(df[LABEL_COL], errors="coerce").notna()].copy()
        labeled[LABEL_COL] = pd.to_numeric(labeled[LABEL_COL]).astype(int)

        examples: list[tuple[str, int, str | None]] = []
        for r in labeled.to_dict("records"):
            text = _row_text(r)
            if not text:
                continue
            opp_id = r.get("opportunity_id")
            opp_id = None if pd.isna(opp_id) or opp_id == "" else str(opp_id)
            examples.append((text, int(r[LABEL_COL]), opp_id))
        return examples

    @cached_property
    def _pool(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str | None], list[str]]:
        """Combined (embeddings, labels, weights, opportunity_ids, texts) pool."""
        seed_texts = [t for t, _ in SEED_EXAMPLES]
        seed_labels = [lbl for _, lbl in SEED_EXAMPLES]
        seed_opp_ids: list[str | None] = [None] * len(SEED_EXAMPLES)
        seed_weights = [SEED_WEIGHT] * len(SEED_EXAMPLES)

        real = self._load_real_examples()
        real_texts = [t for t, _, _ in real]
        real_labels = [lbl for _, lbl, _ in real]
        real_opp_ids = [oid for _, _, oid in real]
        real_weights = [REAL_WEIGHT] * len(real)

        texts = seed_texts + real_texts
        labels = np.array(seed_labels + real_labels, dtype=float)
        weights = np.array(seed_weights + real_weights, dtype=float)
        opp_ids = seed_opp_ids + real_opp_ids

        if real_texts:
            embeddings = np.vstack([
                self._seed_embeddings,
                self._model.encode(real_texts, normalize_embeddings=True),
            ])
        else:
            embeddings = self._seed_embeddings

        return embeddings, labels, weights, opp_ids, texts

    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Score one RFP row by weighted similarity to the labeled example pool.

        relevance_score = 0.0-1.0  → weighted fraction of the k most similar
                                      labeled examples that are positive
        nearest_label/nearest_example/nearest_similarity → the single closest
                                      labeled example, for review/explainability
        """
        embeddings, labels, weights, opp_ids, texts = self._pool
        text = _row_text(row)
        if not text:
            return {
                **row,
                "relevance_score": 0.0,
                "nearest_label": None,
                "nearest_example": None,
                "nearest_similarity": None,
            }

        embedding = self._model.encode(text, normalize_embeddings=True)
        sims = embeddings @ embedding

        # Exclude this row's own labeled entry from its own neighbourhood
        # (otherwise a labeled row trivially matches itself at sim=1.0).
        raw_id = row.get("opportunity_id")
        row_opp_id = None if raw_id is None or pd.isna(raw_id) or raw_id == "" else str(raw_id)
        if row_opp_id is not None:
            keep = np.array([oid != row_opp_id for oid in opp_ids])
        else:
            keep = np.ones(len(sims), dtype=bool)

        idx = np.nonzero(keep)[0]
        order = idx[np.argsort(-sims[idx])]
        top = order[: min(K_NEIGHBORS, len(order))]

        top_sims = sims[top]
        top_labels = labels[top]
        top_weights = weights[top] * np.clip(top_sims, 0.0, None)

        total_weight = float(top_weights.sum())
        final_score = float((top_weights * top_labels).sum() / total_weight) if total_weight > 0 else 0.0

        best = top[0]
        nearest_example = texts[best]
        if len(nearest_example) > NEAREST_EXAMPLE_MAXLEN:
            nearest_example = nearest_example[:NEAREST_EXAMPLE_MAXLEN].rstrip() + "..."

        return {
            **row,
            "relevance_score": round(final_score, 4),
            "nearest_label": int(labels[best]),
            "nearest_example": nearest_example,
            "nearest_similarity": round(float(sims[best]), 4),
        }

    def score_all(
        self,
        rows: list[dict[str, Any]],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[list[dict], list[dict]]:
        """
        Score a list of rows. Returns (flagged, all_scored) where flagged is
        rows with relevance_score >= threshold. All rows stay visible in
        all_scored regardless of score — there is no hard veto.
        """
        scored = [self.score(r) for r in rows]
        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        flagged = [r for r in scored if r["relevance_score"] >= threshold]
        return flagged, scored
