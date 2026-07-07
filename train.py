"""
Fit the clustering pipeline on the TRAIN split and persist everything predict.py
needs. Nothing here is bespoke feature logic; it all goes through pipeline.py so
train and predict stay in lockstep. Run: python train.py
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import pipeline as P


def _internal_metrics(X: np.ndarray, labels: np.ndarray, sample: int = 15000) -> dict:
    # silhouette is O(n^2), so score it on a fixed random sample; the other two
    # are cheap enough to run on everything.
    rng = np.random.RandomState(P.RANDOM_STATE)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    return {
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def main() -> None:
    t0 = time.time()
    train, _eval, _live = P.time_split(P.clean(P.load_raw()))
    print(f"train split: {len(train):,} rows")

    fp = P.FeaturePipeline()
    Xtr = fp.fit_transform(train)
    print(f"feature matrix: {Xtr.shape}")

    # n_init picks the best of several seedings; MiniBatch keeps it cheap at 210k.
    km = MiniBatchKMeans(
        n_clusters=P.N_CLUSTERS, random_state=P.RANDOM_STATE, n_init=10, batch_size=4096
    )
    labels = km.fit_predict(Xtr)

    metrics = _internal_metrics(Xtr, labels)
    print("train internal metrics:", {k: round(v, 4) for k, v in metrics.items()})

    P.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(fp, P.ARTIFACT_DIR / "feature_pipeline.joblib")
    joblib.dump(km, P.ARTIFACT_DIR / "kmeans.joblib")
    config = {
        "n_clusters": P.N_CLUSTERS,
        "svd_components": fp.svd_components,
        "random_state": P.RANDOM_STATE,
        "n_train": int(len(train)),
        "feature_dim": int(Xtr.shape[1]),
        "train_metrics": metrics,
        "cluster_sizes": {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))},
    }
    with open(P.ARTIFACT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"artifacts written to {P.ARTIFACT_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
