"""Cluster stability across KMeans seeds.

The report fixes random_state=42 for reproducibility but the honest question is
whether the k=6 partition is a property of the data or of one lucky init. Fit the
features once (so only the clustering seed varies), refit MiniBatchKMeans at
several seeds, and measure pairwise Adjusted Rand Index. Given silhouette < 0.08
some label churn is expected; ARI quantifies how much. Run: python experiments/seed_stability.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P  # noqa: E402

SEEDS = [0, 1, 7, 42, 123]


def main() -> None:
    train, _eval, _live = P.time_split(P.clean(P.load_raw()))
    fp = P.FeaturePipeline()
    X = fp.fit_transform(train)   # fixed once; only the KMeans seed changes below
    print(f"features: {X.shape}, k={P.N_CLUSTERS}, seeds={SEEDS}")

    labels = {}
    for s in SEEDS:
        km = MiniBatchKMeans(n_clusters=P.N_CLUSTERS, random_state=s,
                             n_init=10, batch_size=4096)
        labels[s] = km.fit_predict(X)

    aris = [adjusted_rand_score(labels[a], labels[b]) for a, b in combinations(SEEDS, 2)]
    for (a, b), v in zip(combinations(SEEDS, 2), aris):
        print(f"  ARI seed {a:>3} vs {b:>3}: {v:.3f}")
    print(f"pairwise ARI  mean {np.mean(aris):.3f}  min {np.min(aris):.3f}  "
          f"max {np.max(aris):.3f}")


if __name__ == "__main__":
    main()
