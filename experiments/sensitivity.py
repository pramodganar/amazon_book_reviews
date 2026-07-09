"""Sensitivity of the k=6 partition to the text-pipeline hyperparameters.

Closes the two remaining open questions from DECISIONS.md:
  1. SVD dimensionality: sweep 50/100/200/400 components on the fixed TF-IDF
     matrix and report retained variance, silhouette, and agreement with the
     shipped labels.
  2. TF-IDF settings: vary min_df, max_df, and the vocabulary cap one at a time
     from the shipped values and measure how much the partition moves.

Reading the ARI column: refitting the shipped configuration at a different seed
already gives ~0.32 (experiments/seed_stability.py), so a variant scoring around
that baseline moves the partition no more than init noise does; markedly lower
means the setting genuinely matters. The numeric block is reused from the shipped
pipeline so only the text side varies. Run: python experiments/sensitivity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P  # noqa: E402

SIL_SAMPLE = 15000
SVD_DIMS = [50, 100, 200, 400]
# One knob at a time from the shipped (max_features=20k, min_df=5, max_df=0.5).
TFIDF_VARIANTS = [
    {"min_df": 2}, {"min_df": 20},
    {"max_df": 0.3}, {"max_df": 0.8},
    {"max_features": 10_000}, {"max_features": 40_000},
]


def _stopwords() -> list[str]:
    return list(ENGLISH_STOP_WORDS | P._CONTRACTION_FRAGMENTS | P._ENTITY_FRAGMENTS)


def _score(X: np.ndarray, base: np.ndarray, labels: np.ndarray,
           rng: np.random.RandomState) -> str:
    idx = rng.choice(len(X), min(SIL_SAMPLE, len(X)), replace=False)
    sil = silhouette_score(X[idx], labels[idx])
    shares = np.bincount(labels, minlength=P.N_CLUSTERS) / len(labels)
    return (f"sil {sil:.4f}  ARI vs shipped {adjusted_rand_score(base, labels):.3f}  "
            f"shares min {shares.min():5.1%} max {shares.max():5.1%}")


def main() -> None:
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    km = joblib.load(P.ARTIFACT_DIR / "kmeans.joblib")
    train, _eval, _live = P.time_split(P.clean(P.load_raw()))
    cleaned = P.clean_text(train["review/text"])

    # Fixed across every variant: the shipped numeric block and labels.
    num_scaled = fp.scaler_.transform(fp._prep_numeric(train))
    base = km.predict(fp.transform(train))
    rng = np.random.RandomState(P.RANDOM_STATE)

    def cluster(text_svd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        text_scaled = StandardScaler().fit_transform(text_svd)
        text = text_scaled / np.sqrt(text_scaled.shape[1])
        num = num_scaled / np.sqrt(num_scaled.shape[1])
        X = np.hstack([text, num]).astype("float32")
        labels = MiniBatchKMeans(n_clusters=P.N_CLUSTERS, random_state=P.RANDOM_STATE,
                                 n_init=10, batch_size=4096).fit_predict(X)
        return X, labels

    print("[1. SVD dimensionality sweep, shipped TF-IDF]")
    tfidf = fp.vectorizer_.transform(cleaned)
    for d in SVD_DIMS:
        t0 = time.time()
        svd = TruncatedSVD(n_components=d, random_state=P.RANDOM_STATE)
        text_svd = svd.fit_transform(tfidf)
        X, labels = cluster(text_svd)
        print(f"  d={d:>3}  var {svd.explained_variance_ratio_.sum():5.1%}  "
              f"{_score(X, base, labels, rng)}  ({time.time() - t0:.0f}s)")

    print("\n[2. TF-IDF variants, SVD 200]")
    for override in TFIDF_VARIANTS:
        t0 = time.time()
        params = dict(max_features=P.TFIDF_MAX_FEATURES, min_df=P.TFIDF_MIN_DF,
                      max_df=P.TFIDF_MAX_DF, stop_words=_stopwords(), sublinear_tf=True)
        params.update(override)
        vec = TfidfVectorizer(**params)
        m = vec.fit_transform(cleaned)
        svd = TruncatedSVD(n_components=P.SVD_COMPONENTS, random_state=P.RANDOM_STATE)
        X, labels = cluster(svd.fit_transform(m))
        name = ", ".join(f"{k}={v}" for k, v in override.items())
        print(f"  {name:<20} vocab {m.shape[1]:>6,}  "
              f"{_score(X, base, labels, rng)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
