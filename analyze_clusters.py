"""
Generate the cluster interpretation artifact (artifacts/cluster_meta.json) used by
the Streamlit app and the report: human labels, sizes, top distinctive terms,
numeric centroid profiles, and ANOVA feature importance. Run after train.py.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE

import pipeline as P

# How many points to embed for the 2D cluster map. t-SNE is O(n^2)-ish; a couple
# thousand is plenty to show the shape (and honestly, the overlap) of the segments.
SCATTER_SAMPLE = 2500

# Human labels written from the cluster analysis. Kept here as the one source of
# truth; the app reads them back out of the generated JSON.
LABELS = {
    0: "How-to / reference reviewer",
    1: "Punchy popular-book reactor",
    2: "Prolific plot-summarizer",
    3: "General opinion reviewer",
    4: "Argumentative deep-diver",
    5: "Analytical critic / essayist",
}


def main() -> None:
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    km = joblib.load(P.ARTIFACT_DIR / "kmeans.joblib")

    train, ev, live = P.time_split(P.clean(P.load_raw()))
    X = fp.transform(train)
    labels = km.predict(X)

    # top distinctive terms: cluster mean tf-idf minus the global mean
    tfidf = fp.vectorizer_.transform(P.clean_text(train["review/text"]))
    terms = np.array(fp.vectorizer_.get_feature_names_out())
    overall = np.asarray(tfidf.mean(axis=0)).ravel()

    num = fp.transform_numeric(train)
    F, _p = f_classif(num.values, labels)

    meta = {"labels": LABELS, "clusters": {}, "anova": {}}
    for c in range(P.N_CLUSTERS):
        mask = labels == c
        cmean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top = terms[np.argsort(cmean - overall)[::-1][:12]].tolist()
        meta["clusters"][str(c)] = {
            "label": LABELS[c],
            "size": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "top_terms": top,
            "profile": {k: round(float(v), 2) for k, v in num[mask].mean().items()},
        }
    meta["anova"] = {k: round(float(v), 1)
                     for k, v in sorted(zip(num.columns, F), key=lambda t: -t[1])}

    # Cluster-share drift across the time-ordered splits. Train is reused from
    # above; eval/live are transform-only (the whole point of the shared pipeline).
    def shares(labs):
        b = np.bincount(labs, minlength=P.N_CLUSTERS) / len(labs)
        return {str(c): round(float(b[c]), 4) for c in range(P.N_CLUSTERS)}

    meta["drift"] = {
        "train": shares(labels),
        "eval": shares(km.predict(fp.transform(ev))),
        "live": shares(km.predict(fp.transform(live))),
    }

    with open(P.ARTIFACT_DIR / "cluster_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", P.ARTIFACT_DIR / "cluster_meta.json")

    # 2D t-SNE projection for the app's cluster map. Purely for visualization:
    # t-SNE distances aren't meaningful, but it conveys the (overlapping) shape.
    rng = np.random.RandomState(P.RANDOM_STATE)
    idx = rng.choice(len(X), min(SCATTER_SAMPLE, len(X)), replace=False)
    emb = TSNE(n_components=2, random_state=P.RANDOM_STATE,
               perplexity=30, init="pca").fit_transform(X[idx])
    snippet = (train["review/text"].iloc[idx].astype(str)
               .str.slice(0, 90).str.replace(r"\s+", " ", regex=True))
    scatter = pd.DataFrame({
        "x": emb[:, 0], "y": emb[:, 1],
        "cluster": labels[idx],
        "label": [LABELS[c] for c in labels[idx]],
        "snippet": snippet.to_numpy(),
    })
    scatter.to_parquet(P.ARTIFACT_DIR / "scatter.parquet")
    print("wrote", P.ARTIFACT_DIR / "scatter.parquet")


if __name__ == "__main__":
    main()
