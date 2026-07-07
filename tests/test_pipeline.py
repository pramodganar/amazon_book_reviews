"""Pipeline invariants that don't need the 260MB dataset.

These build tiny in-memory frames and assert the contracts the report leans on:
preprocessing correctness, leak-free/parity behaviour, output shape, unseen-vocab
handling, and seed reproducibility. Run: pytest -q
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

import pipeline as P


# Pool with enough distinct terms that no word lands in >50% of docs (max_df=0.5)
# yet each still clears min_df=5 once there are ~60 rows. Uniform text would give
# TfidfVectorizer an empty vocabulary, so tests that fit TF-IDF use _reviews().
_POOL = ["great", "boring", "novel", "history", "fun", "dull", "classic",
         "thriller", "romance", "science", "war", "family", "mystery", "poetry"]


def _reviews(n):
    return [f"this book was really {_POOL[i % len(_POOL)]} and quite "
            f"{_POOL[(i * 3 + 1) % len(_POOL)]} overall" for i in range(n)]


def _frame(texts, ids=None, users=None, times=None):
    n = len(texts)
    return pd.DataFrame({
        "Id": ids if ids is not None else [f"B{i}" for i in range(n)],
        "Title": ["t"] * n,
        "User_id": users if users is not None else [f"U{i}" for i in range(n)],
        "profileName": ["p"] * n,
        "review/time": times if times is not None else [1350000000 + i for i in range(n)],
        "review/text": texts,
    })


def test_clean_text_strips_html_entities_and_urls():
    s = pd.Series(["Great book &amp; a <b>must</b> read http://x.com !!!"])
    out = P.clean_text(s).iloc[0]
    assert "amp" not in out.split() and "http" not in out
    assert "great" in out and "book" in out and "must" in out


def test_clean_text_keeps_content_drops_apostrophes():
    # clean_text keeps only letters/spaces; contraction fragments ("ve","ll") are
    # dropped later by the vectorizer stopword list, not here.
    out = P.clean_text(pd.Series(["I've read it, you'll love it"])).iloc[0]
    assert "read" in out and "love" in out
    assert "'" not in out and "," not in out


def test_structural_features_use_raw_case():
    # upper_ratio / exclaim must see original casing and punctuation, because they
    # are computed before any lowercasing.
    f = P._structural_features(_frame(["THIS IS GREAT!!"]))
    assert f["upper_ratio"].iloc[0] > 0.9
    assert f["exclaim_count"].iloc[0] == 2


def test_structural_features_guard_no_div_by_zero():
    # whitespace-only -> 0 words (avg_word_len guard); pure punctuation -> 0 letters
    # (upper_ratio guard). Nothing should be NaN/inf.
    f = P._structural_features(_frame(["   ", "...!!!"]))
    assert np.isfinite(f.to_numpy()).all()
    assert f["avg_word_len"].iloc[0] == 0.0     # whitespace row, word_count == 0
    assert f["upper_ratio"].iloc[1] == 0.0      # punctuation row, no letters
    assert f["exclaim_count"].iloc[1] == 3


def test_clean_drops_exact_duplicates_and_bad_timestamps():
    df = _frame(
        ["same review text", "same review text", "a fine review text"],
        ids=["B0", "B0", "B2"],
        users=["U0", "U0", "U2"],
        times=[1350000000, 1350000000, -1],     # rows 0/1 identical; row 2 sentinel
    )
    out = P.clean(df)
    assert len(out) == 1                          # one dup dropped, one bad-time dropped


def test_fit_transform_shape_and_finite():
    train = _frame(_reviews(60))
    fp = P.FeaturePipeline(svd_components=5)       # small so it fits a tiny corpus
    X = fp.fit_transform(train)
    assert X.shape[0] == 60
    assert np.isfinite(X).all()


def test_transform_handles_unseen_vocab_and_user():
    fp = P.FeaturePipeline(svd_components=5)
    fp.fit_transform(_frame(_reviews(60)))
    live = _frame(["zzzqqq unheard vocabulary token"], ids=["NEW_B"], users=["NEW_USER"])
    X = fp.transform(P.clean(live))
    assert np.isfinite(X).all()                   # all-unseen vocab must not break it
    act = fp._activity_features(P.clean(live))
    assert act["user_review_count"].iloc[0] == 1.0   # unseen user -> count 1


def test_seed_reproducibility_and_label_range():
    fp = P.FeaturePipeline(svd_components=5)
    X = fp.fit_transform(_frame(_reviews(120)))
    a = MiniBatchKMeans(n_clusters=3, random_state=P.RANDOM_STATE, n_init=5).fit_predict(X)
    b = MiniBatchKMeans(n_clusters=3, random_state=P.RANDOM_STATE, n_init=5).fit_predict(X)
    assert (a == b).all()                         # same seed -> identical labels
    assert set(np.unique(a)).issubset({0, 1, 2})  # labels in 0..k-1, no NaN
