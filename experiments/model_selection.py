"""Model-selection evidence behind the k=6 choice.

Reproduces the numbers cited in reports/report.md section 4 and regenerates
reports/figures/k_selection.png. Three parts:
  1. MiniBatchKMeans swept over k, scored on inertia (elbow), silhouette,
     Davies-Bouldin, Calinski-Harabasz -> the figure and the k table.
  2. DBSCAN on a subsample across several eps -> the "no density structure"
     negative result, plus the 5th-NN distance spread.
  3. Agglomerative on a subsample across k -> the silhouette sanity check.

Reuses the fitted pipeline from train.py so the train matrix is identical to the
one the shipped model was fit on. Run: python experiments/model_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P  # noqa: E402

K_RANGE = range(2, 16)          # k = 2..15
SIL_SAMPLE = 15000              # silhouette is O(n^2); score on a fixed sample
DBSCAN_SAMPLE = 20000
AGG_SAMPLE = 5000               # Agglomerative is O(n^2) in memory
DBSCAN_EPS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
FIG_PATH = P.PROJECT_DIR / "reports" / "figures" / "k_selection.png"


def _train_matrix() -> np.ndarray:
    """Train feature matrix, reusing the shipped fitted pipeline."""
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    train, _eval, _live = P.time_split(P.clean(P.load_raw()))
    return fp.transform(train)


def sweep_k(X: np.ndarray) -> dict:
    rng = np.random.RandomState(P.RANDOM_STATE)
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    rows = {"k": [], "inertia": [], "silhouette": [], "davies_bouldin": [],
            "calinski_harabasz": []}
    for k in K_RANGE:
        km = MiniBatchKMeans(n_clusters=k, random_state=P.RANDOM_STATE,
                             n_init=10, batch_size=4096)
        labels = km.fit_predict(X)
        rows["k"].append(k)
        rows["inertia"].append(float(km.inertia_))
        rows["silhouette"].append(float(silhouette_score(X[idx], labels[idx])))
        rows["davies_bouldin"].append(float(davies_bouldin_score(X, labels)))
        rows["calinski_harabasz"].append(float(calinski_harabasz_score(X, labels)))
        print(f"  k={k:>2}  inertia={rows['inertia'][-1]:>12.0f}  "
              f"sil={rows['silhouette'][-1]:.4f}  "
              f"db={rows['davies_bouldin'][-1]:.3f}  "
              f"ch={rows['calinski_harabasz'][-1]:.0f}")
    return rows


def probe_dbscan(X: np.ndarray) -> None:
    rng = np.random.RandomState(P.RANDOM_STATE)
    sample = X[rng.choice(len(X), min(DBSCAN_SAMPLE, len(X)), replace=False)]

    nn = NearestNeighbors(n_neighbors=5).fit(sample)
    d5 = nn.kneighbors(sample)[0][:, -1]
    print(f"  5th-NN distance: min {d5.min():.2f}  median {np.median(d5):.2f}  "
          f"max {d5.max():.2f}  (p10 {np.percentile(d5, 10):.2f}, "
          f"p90 {np.percentile(d5, 90):.2f})")
    for eps in DBSCAN_EPS:
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(sample)
        n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        biggest = 0.0
        if n_clusters:
            counts = np.bincount(lab[lab >= 0])
            biggest = counts.max() / len(sample)
        print(f"  eps={eps:<4}  clusters={n_clusters:<3}  noise={noise:6.1%}  "
              f"largest_cluster={biggest:6.1%}")


def probe_agglomerative(X: np.ndarray) -> None:
    rng = np.random.RandomState(P.RANDOM_STATE)
    sample = X[rng.choice(len(X), min(AGG_SAMPLE, len(X)), replace=False)]
    for k in (2, 4, 6, 8, 10):
        lab = AgglomerativeClustering(n_clusters=k).fit_predict(sample)
        sil = silhouette_score(sample, lab)
        counts = np.bincount(lab)
        print(f"  k={k:>2}  silhouette={sil:+.4f}  "
              f"sizes={sorted(counts.tolist(), reverse=True)}")


def plot_k_selection(rows: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = rows["k"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in (
        (axes[0], "inertia", "Inertia (elbow)"),
        (axes[1], "silhouette", "Silhouette (higher is better)"),
        (axes[2], "davies_bouldin", "Davies-Bouldin (lower is better)"),
    ):
        ax.plot(k, rows[key], marker="o")
        ax.axvline(P.N_CLUSTERS, color="crimson", ls="--", lw=1, label=f"k={P.N_CLUSTERS}")
        ax.set_xlabel("k")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("k-selection sweep (MiniBatchKMeans on the 210-dim train matrix)")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    print(f"wrote {FIG_PATH}")


def main() -> None:
    X = _train_matrix()
    print(f"train matrix: {X.shape}\n[k sweep]")
    rows = sweep_k(X)
    print("\n[DBSCAN probe]")
    probe_dbscan(X)
    print("\n[Agglomerative probe]")
    probe_agglomerative(X)
    print()
    plot_k_selection(rows)


if __name__ == "__main__":
    main()
