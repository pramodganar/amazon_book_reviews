"""
Assign cluster labels to new reviews using the artifacts fit by train.py. The
same pipeline.transform path runs here, so live rows get byte-for-byte identical
feature steps, the whole point of the shared module.

Usage:
    python predict.py                      # scores the held-out live split
    python predict.py path/to/reviews.csv  # scores an arbitrary file
    python predict.py path/to/reviews.csv --out labelled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

import pipeline as P


def load_artifacts():
    fp = joblib.load(P.ARTIFACT_DIR / "feature_pipeline.joblib")
    km = joblib.load(P.ARTIFACT_DIR / "kmeans.joblib")
    return fp, km


def assign(df: pd.DataFrame, fp: P.FeaturePipeline, km) -> pd.DataFrame:
    """Transform -> nearest centroid. Expects already-cleaned rows.

    Cleaning happens in main so it runs exactly once on either input path (the
    live split is cleaned before it can be split; a passed file is cleaned when
    loaded).
    """
    X = fp.transform(df)
    out = df.copy()
    out["cluster"] = km.predict(X)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None,
                    help="CSV/DB of reviews to score. Defaults to the live split.")
    ap.add_argument("--out", default=None, help="Optional path to write labelled rows.")
    args = ap.parse_args()

    fp, km = load_artifacts()

    if args.source is None:
        # No file given: reproduce the live split so it runs out of the box.
        _train, _eval, df = P.time_split(P.clean(P.load_raw()))
    else:
        df = P.clean(P.load_raw(args.source))

    labelled = assign(df, fp, km)
    dist = labelled["cluster"].value_counts().sort_index()
    print("cluster distribution:")
    print(dist.to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        labelled.to_csv(args.out, index=False)
        print(f"wrote {len(labelled):,} labelled rows to {args.out}")


if __name__ == "__main__":
    main()
