"""Ablations behind the open questions in reports/report.md sections 6 and 7.

Four checks, all on the train matrix from the shipped fitted pipeline (so the
features are identical to the model's):
  1. Full KMeans vs the shipped MiniBatchKMeans: is the mini-batch approximation
     a contributor to the low seed-ARI, or is it the data?
  2. dow ablation: refit k=6 without day-of-week and measure how much the
     partition moves, against the seed-churn baseline (~0.32 ARI).
  3. Text/numeric block-weight sweep: does tilting toward topic or style beat the
     equal-total-variance default?
  4. Per-era silhouette: is eval scoring above train explained by the eval window
     being a narrower, more homogeneous era?

Rebuilds the two standardized blocks once via the pipeline's internals (the
private methods are used deliberately: the point is to reuse the exact fitted
state, not refit it). Run: python experiments/ablations.py
"""

from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P  # noqa: E402

SIL_SAMPLE = 15000                  # silhouette is O(n^2); score on a fixed sample
FULL_KMEANS_SEEDS = [0, 1, 42]      # pairwise ARI among full fits isolates the data
WEIGHTS = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]  # (text, numeric)


def _sil(X: np.ndarray, labels: np.ndarray, rng: np.random.RandomState) -> float:
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def _fit_minibatch(X: np.ndarray, seed: int = P.RANDOM_STATE) -> np.ndarray:
    return MiniBatchKMeans(n_clusters=P.N_CLUSTERS, random_state=seed,
                           n_init=10, batch_size=4096).fit_predict(X)


def main() -> None:
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    km = joblib.load(P.ARTIFACT_DIR / "kmeans.joblib")
    train, _eval, _live = P.time_split(P.clean(P.load_raw()))

    # The two standardized blocks, once; hstacking them the way _combine does
    # reproduces fp.transform(train) exactly.
    text_scaled = fp.text_scaler_.transform(fp._transform_text(train))
    num_prepped = fp._prep_numeric(train)
    num_scaled = fp.scaler_.transform(num_prepped)

    def combine(tw: float = 1.0, nw: float = 1.0, num: np.ndarray = num_scaled) -> np.ndarray:
        text = text_scaled / np.sqrt(text_scaled.shape[1]) * tw
        n = num / np.sqrt(num.shape[1]) * nw
        return np.hstack([text, n]).astype("float32")

    X = combine()
    base = km.predict(X)
    rng = np.random.RandomState(P.RANDOM_STATE)
    print(f"train matrix: {X.shape}; shipped-model silhouette {_sil(X, base, rng):.4f}")

    print("\n[1. full KMeans vs shipped MiniBatchKMeans]")
    full_labels = {}
    for seed in FULL_KMEANS_SEEDS:
        t0 = time.time()
        full_labels[seed] = KMeans(n_clusters=P.N_CLUSTERS, random_state=seed,
                                   n_init=10).fit_predict(X)
        print(f"  seed {seed:>2}: fit {time.time() - t0:5.1f}s  "
              f"ARI vs shipped {adjusted_rand_score(base, full_labels[seed]):.3f}  "
              f"sil {_sil(X, full_labels[seed], rng):.4f}")
    aris = [adjusted_rand_score(full_labels[a], full_labels[b])
            for a, b in combinations(FULL_KMEANS_SEEDS, 2)]
    print(f"  pairwise ARI among full-KMeans seeds: mean {np.mean(aris):.3f}  "
          f"min {np.min(aris):.3f}  max {np.max(aris):.3f}")

    print("\n[2. dow ablation]")
    dow_idx = P.NUMERIC_FEATURES.index("dow")
    num_nodow = np.delete(num_prepped, dow_idx, axis=1)
    from sklearn.preprocessing import StandardScaler
    num_nodow = StandardScaler().fit_transform(num_nodow)
    X_nodow = combine(num=num_nodow)
    lab_nodow = _fit_minibatch(X_nodow)
    print(f"  ARI vs shipped: {adjusted_rand_score(base, lab_nodow):.3f} "
          f"(seed-churn baseline ~0.32)  sil {_sil(X_nodow, lab_nodow, rng):.4f}")
    # persona persistence: how concentrated is each ablated cluster in one shipped cluster
    for c in range(P.N_CLUSTERS):
        mask = lab_nodow == c
        overlap = np.bincount(base[mask], minlength=P.N_CLUSTERS) / mask.sum()
        j = int(np.argmax(overlap))
        print(f"  ablated c{c} ({mask.mean():5.1%}) -> shipped c{j} at {overlap[j]:5.1%}")

    print("\n[3. block-weight sweep]")
    for tw, nw in WEIGHTS:
        Xw = combine(tw, nw)
        lab = _fit_minibatch(Xw)
        shares = np.bincount(lab, minlength=P.N_CLUSTERS) / len(lab)
        print(f"  text={tw} num={nw}:  sil {_sil(Xw, lab, rng):.4f}  "
              f"ARI vs shipped {adjusted_rand_score(base, lab):.3f}  "
              f"shares min {shares.min():.1%} max {shares.max():.1%}")

    print("\n[4. per-era silhouette on train]")
    n_eval = 59765  # eval-split size; the last train rows form a comparably narrow window
    late = np.arange(len(X) - n_eval, len(X))
    idx_all = rng.choice(len(X), SIL_SAMPLE, replace=False)
    idx_late = late[rng.choice(len(late), SIL_SAMPLE, replace=False)]
    print(f"  full train era (1995-2009): {silhouette_score(X[idx_all], base[idx_all]):.4f}")
    print(f"  late-train window (last {n_eval:,} rows): "
          f"{silhouette_score(X[idx_late], base[idx_late]):.4f}")
    print("  (eval split scores 0.051 vs train 0.034 in evaluate.py; if the narrow "
          "window scores above the full era, era homogeneity explains the gap)")


if __name__ == "__main__":
    main()
