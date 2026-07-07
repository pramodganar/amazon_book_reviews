"""Out-of-sample evaluation behind reports/report.md section 6.

Scores the shipped model on all three time-ordered splits with the fitted
centroids (no refit) and reports:
  1. Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) per split.
  2. Distance-to-centroid drift: mean nearest-centroid distance per split and the
     live distribution (median/max), the anomaly-detection hook.
Everything stateful comes from train.py's artifacts; eval/live are transform-only.
Run: python evaluate.py
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import pipeline as P

SIL_SAMPLE = 15000  # silhouette is O(n^2); score on a fixed sample


def _metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    rng = np.random.RandomState(P.RANDOM_STATE)
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    return {
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def main() -> None:
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    km = joblib.load(P.ARTIFACT_DIR / "kmeans.joblib")
    train, ev, live = P.time_split(P.clean(P.load_raw()))

    splits = {"train": train, "eval": ev, "live": live}
    Xs, labels, dmin = {}, {}, {}
    for name, df in splits.items():
        X = fp.transform(df)
        Xs[name] = X
        labels[name] = km.predict(X)
        dmin[name] = km.transform(X).min(axis=1)  # nearest-centroid distance

    print("internal metrics per split:")
    print(f"{'split':<6} {'silhouette':>11} {'davies_b':>9} {'calinski_h':>11}")
    for name in splits:
        m = _metrics(Xs[name], labels[name])
        print(f"{name:<6} {m['silhouette']:>11.3f} {m['davies_bouldin']:>9.2f} "
              f"{m['calinski_harabasz']:>11.0f}")

    print("\nnearest-centroid distance (drift / anomaly hook):")
    base = dmin["train"].mean()
    print(f"{'split':<6} {'mean':>7} {'median':>7} {'max':>7} {'d_mean_vs_train':>16}")
    for name in splits:
        d = dmin[name]
        print(f"{name:<6} {d.mean():>7.2f} {np.median(d):>7.2f} {d.max():>7.2f} "
              f"{d.mean() - base:>+16.2f}")

    print("\nper-cluster mean nearest-centroid distance (train -> live):")
    for c in range(P.N_CLUSTERS):
        tr = dmin["train"][labels["train"] == c].mean()
        lv_mask = labels["live"] == c
        lv = dmin["live"][lv_mask].mean() if lv_mask.any() else float("nan")
        print(f"  c{c}: {tr:.2f} -> {lv:.2f}  (delta {lv - tr:+.2f})")


if __name__ == "__main__":
    main()
