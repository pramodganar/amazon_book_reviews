"""
Shared data pipeline for the Amazon Book Reviews clustering project.

Everything that touches the data lives here so that train.py and predict.py run
the exact same steps: loading, cleaning, and feature engineering. Train fits the
stateful objects (vectorizer, SVD, scalers, activity counts); predict only
transforms. Anything that fits state exposes both a fit path and a transform path.
"""

from __future__ import annotations

import html
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# --- config -----------------------------------------------------------------
# Kept as plain module constants for now. If this grows I'd move it to a small
# config object, but for three call sites that's premature.

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / "raw" / "amazon_book_reviews.csv"
DB_PATH = DATA_DIR / "raw" / "clustering.db"
DB_TABLE = "amazon_book_reviews"

# Where train.py writes and predict.py reads the fitted objects.
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

# MiniBatchKMeans with k=6 gives balanced, interpretable segments. Internal
# metrics favour smaller k, but k=2 is degenerate for segmentation; see the
# report for the full justification.
N_CLUSTERS = 6

# The six columns that actually exist in the data. The Data Card advertises ten
# (Price, review/helpfulness, review/score, review/summary), but none of those
# are present in the CSV, so we never reference them anywhere.
EXPECTED_COLUMNS = ["Id", "Title", "User_id", "profileName", "review/time", "review/text"]

# Time-ordered split. Sorting by review/time mimics production: we fit on the
# past and score the future, so a cluster model can't peek at reviews that were
# written after the ones it's scoring.
SPLIT_FRACTIONS = (0.70, 0.20, 0.10)  # train / eval / live

# Below this many non-whitespace characters a review carries no usable signal.
MIN_TEXT_CHARS = 3

# Reproducibility for every stochastic sklearn step (SVD, KMeans, samplers).
RANDOM_STATE = 42

# TF-IDF: cap the vocabulary so the matrix stays manageable and the SVD is cheap.
# min_df drops near-unique terms (typos, ids); max_df drops terms so common they
# don't separate anyone. sublinear_tf dampens the effect of very long reviews.
TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.5

# TruncatedSVD (LSA) target. Raw TF-IDF is ~20k-dim and sparse, useless for
# distance-based clustering. 200 dims keeps most of the signal while making
# KMeans tractable; we report the retained variance to justify the number.
SVD_COMPONENTS = 200

# Numeric features that are heavily right-skewed / power-law. log1p pulls the
# long tail in so a few 30k-char reviews or 500-review users don't dominate the
# Euclidean distance. upper_ratio (bounded 0-1), dow and year are left alone.
LOG1P_FEATURES = [
    "char_len", "word_count", "exclaim_count", "question_count",
    "user_review_count", "product_review_count",
]
# A pure-punctuation review can report an absurd avg_word_len; cap it.
AVG_WORD_LEN_CLIP = 15.0


# --- loading ----------------------------------------------------------------

def load_raw(source: str | Path | None = None) -> pd.DataFrame:
    """Load the raw reviews table.

    Prefers the SQLite DB (that's the stated source of truth); falls back to the
    CSV when the DB isn't present, which is the current situation on disk.
    """
    if source is not None:
        source = Path(source)
        if source.suffix == ".db":
            return _load_sqlite(source)
        return pd.read_csv(source)

    if DB_PATH.exists():
        return _load_sqlite(DB_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(
        f"No data found. Looked for {DB_PATH} and {CSV_PATH}."
    )


def _load_sqlite(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        return pd.read_sql(f"SELECT * FROM {DB_TABLE}", conn)


# --- cleaning ---------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning shared by train and predict.

    This is deliberately stateless: no counts, means, or vocabularies are
    learned here. That's what lets predict.py clean live data with the identical
    rules and no fitted artifacts. Every rule drops rows that are genuinely
    unusable, and each drop is small relative to 300k.
    """
    n0 = len(df)

    # Guard against a schema drift in whatever source we were handed.
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing expected columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()

    # Exact duplicate reviews. These would otherwise form artificially dense
    # clusters and drag centroids toward whatever text was copy-pasted most.
    df = df.drop_duplicates()

    # review/time is a unix timestamp; negatives/zero are sentinels (the data
    # has a handful of -1s that decode to 1969). They're invalid and, because we
    # split by time, they'd all pile up at the very front of the train set.
    df["review/time"] = pd.to_numeric(df["review/time"], errors="coerce")
    df = df[df["review/time"] > 0]

    # Text is the core clustering signal. Drop rows with no usable text.
    text = df["review/text"].astype("string").str.strip()
    df = df[text.str.len() >= MIN_TEXT_CHARS]

    # Note on null User_id (~19% of rows): we keep these. The text is still
    # valuable, and we deliberately do NOT fill them with a shared placeholder,
    # which would fabricate one giant "user". They're handled at feature time as
    # distinct anonymous reviewers, each with an activity count of 1.

    df = df.reset_index(drop=True)
    print(f"clean: {n0:,} -> {len(df):,} rows ({n0 - len(df):,} dropped)")
    return df


# --- time-ordered split -----------------------------------------------------

def time_split(df: pd.DataFrame, fractions: tuple[float, float, float] = SPLIT_FRACTIONS):
    """Split into train / eval / live by chronological order.

    Sort ascending by review/time, then cut by row position. A stable secondary
    sort on Id makes the boundaries reproducible when many reviews share the same
    day-granular timestamp (they do; timestamps here are day-aligned).
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"

    ordered = df.sort_values(["review/time", "Id"], kind="stable").reset_index(drop=True)
    n = len(ordered)
    i_train = int(n * fractions[0])
    i_eval = int(n * (fractions[0] + fractions[1]))

    train = ordered.iloc[:i_train].copy()
    eval_ = ordered.iloc[i_train:i_eval].copy()
    live = ordered.iloc[i_eval:].copy()
    return train, eval_, live


# --- numeric features -------------------------------------------------------
# Three families, all interpretable so we can profile clusters on them later:
#   structural/tone : shape of the writing (length, shouting, punctuation)
#   temporal        : when it was written
#   activity        : how prolific the reviewer / how reviewed the product is
#
# Ordered column list so the matrix layout is stable across fit and transform.
STRUCTURAL_FEATURES = [
    "char_len", "word_count", "avg_word_len",
    "upper_ratio", "exclaim_count", "question_count",
]
TEMPORAL_FEATURES = ["dow", "year"]  # hour is dropped: every timestamp is midnight-aligned
ACTIVITY_FEATURES = ["user_review_count", "product_review_count"]

NUMERIC_FEATURES = STRUCTURAL_FEATURES + TEMPORAL_FEATURES + ACTIVITY_FEATURES


def _structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tone/shape features from the RAW text.

    Deliberately computed before any lowercasing or punctuation stripping:
    uppercase ratio and exclamation counts (our proxies for shouting/harshness)
    only survive on the original text. All stateless: pure functions of a row.
    """
    s = df["review/text"].astype("string").fillna("")

    char_len = s.str.len()
    word_count = s.str.split().str.len().fillna(0)
    non_space = s.str.replace(r"\s", "", regex=True).str.len()

    letters = s.str.count(r"[A-Za-z]")
    uppers = s.str.count(r"[A-Z]")

    out = pd.DataFrame(index=df.index)
    out["char_len"] = char_len.astype("float64")
    out["word_count"] = word_count.astype("float64")
    # guard the divisions: some rows are pure punctuation ("...") -> 0 words/letters
    out["avg_word_len"] = np.where(word_count > 0, non_space / word_count, 0.0)
    out["upper_ratio"] = np.where(letters > 0, uppers / letters, 0.0)
    out["exclaim_count"] = s.str.count("!").astype("float64")
    out["question_count"] = s.str.count(r"\?").astype("float64")
    return out


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["review/time"], unit="s")
    out = pd.DataFrame(index=df.index)
    out["dow"] = dt.dt.dayofweek.astype("float64")   # 0=Mon..6=Sun
    out["year"] = dt.dt.year.astype("float64")
    return out


_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_NONALPHA_RE = re.compile(r"[^a-z\s]+")

# Fragments left behind when apostrophes are stripped from contractions
# ("I've" -> "ve", "don't" -> "don t"). They're frequent so min_df won't drop
# them, and they carry no topic signal, so we treat them as stopwords.
_CONTRACTION_FRAGMENTS = {"ve", "ll", "re", "s", "t", "m", "d", "don", "didn", "doesn", "isn", "wasn"}
# Remnants of doubly-encoded HTML entities (&amp;lt; survives one unescape pass).
_ENTITY_FRAGMENTS = {"lt", "gt", "amp", "quot", "nbsp", "apos"}


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize review text for the bag-of-words path.

    Order matters: unescape HTML entities first (&quot; -> ", &amp; -> &) so they
    don't survive as junk tokens like "quot"/"amp", then drop tags and URLs, then
    keep only letters and spaces. Stopword removal and tokenization are left to
    the TfidfVectorizer so there's a single place that defines the vocabulary.
    Stateless: predict.py runs the identical function on live text.
    """
    s = series.astype("string").fillna("").str.lower()
    s = s.map(html.unescape)  # &quot; &amp; &#39; ... -> real chars, then stripped below
    s = s.str.replace(_HTML_RE, " ", regex=True)
    s = s.str.replace(_URL_RE, " ", regex=True)
    s = s.str.replace(_NONALPHA_RE, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


class FeaturePipeline:
    """Holds the fitted state and turns cleaned rows into a numeric matrix.

    Structural and temporal features are stateless, but the activity counts are
    learned from train and reused at transform time; that's the whole reason
    this is a stateful object with a fit/transform split rather than a plain
    function. The TF-IDF vectorizer, SVD, and scalers hang off the same object so
    one artifact carries everything predict.py needs.
    """

    def __init__(self, svd_components: int = SVD_COMPONENTS,
                 text_weight: float = 1.0, numeric_weight: float = 1.0):
        # activity state
        self.user_counts_: dict | None = None
        self.product_counts_: dict | None = None
        # text state
        self.vectorizer_: TfidfVectorizer | None = None
        self.svd_: TruncatedSVD | None = None
        self.text_scaler_: StandardScaler | None = None
        # numeric state
        self.scaler_: StandardScaler | None = None
        # block-balance weights (see _combine); tunable so I can shift emphasis
        self.svd_components = svd_components
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight

    # -- activity (stateful, leak-free) --------------------------------------

    def _fit_activity(self, df: pd.DataFrame) -> None:
        # Activity level = how often this reviewer / product appears in TRAIN.
        # Learning it only from train keeps it leak-free: an eval/live review
        # can't inflate its own author's count with future activity.
        self.user_counts_ = df["User_id"].value_counts().to_dict()
        self.product_counts_ = df["Id"].value_counts().to_dict()

    def _activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.user_counts_ is not None, "call fit before transform"
        out = pd.DataFrame(index=df.index)
        # Unseen users (and the ~19% anonymous nulls, which never match) map to 1:
        # the single review we're looking at is the only activity we can attribute.
        out["user_review_count"] = (
            df["User_id"].map(self.user_counts_).fillna(1.0).astype("float64")
        )
        out["product_review_count"] = (
            df["Id"].map(self.product_counts_).fillna(1.0).astype("float64")
        )
        return out

    def transform_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw (unscaled) numeric feature block, ordered by NUMERIC_FEATURES.

        Returned unscaled on purpose: these are the interpretable values used for
        cluster centroid profiles and the ANOVA feature-importance analysis. The
        scaled version used for clustering is produced separately below.
        """
        feats = pd.concat(
            [_structural_features(df), _temporal_features(df), self._activity_features(df)],
            axis=1,
        )
        return feats[NUMERIC_FEATURES]

    # -- text (stateful: TF-IDF + LSA) ---------------------------------------

    def _fit_text(self, df: pd.DataFrame) -> np.ndarray:
        cleaned = clean_text(df["review/text"])
        self.vectorizer_ = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words=list(ENGLISH_STOP_WORDS | _CONTRACTION_FRAGMENTS | _ENTITY_FRAGMENTS),
            sublinear_tf=True,
        )
        tfidf = self.vectorizer_.fit_transform(cleaned)
        self.svd_ = TruncatedSVD(n_components=self.svd_components, random_state=RANDOM_STATE)
        return self.svd_.fit_transform(tfidf)

    def _transform_text(self, df: pd.DataFrame) -> np.ndarray:
        assert self.vectorizer_ is not None and self.svd_ is not None
        tfidf = self.vectorizer_.transform(clean_text(df["review/text"]))
        return self.svd_.transform(tfidf)

    # -- numeric scaling -----------------------------------------------------

    def _prep_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """Raw numeric -> skew-corrected array (pre-scaling), stable column order."""
        num = self.transform_numeric(df).copy()
        num["avg_word_len"] = num["avg_word_len"].clip(upper=AVG_WORD_LEN_CLIP)
        for col in LOG1P_FEATURES:
            num[col] = np.log1p(num[col])
        return num.to_numpy()

    # -- assembly ------------------------------------------------------------

    def _combine(self, text_svd: np.ndarray, num_scaled: np.ndarray) -> np.ndarray:
        """Concatenate the two blocks on equal footing.

        Straight concatenation would let the 200-dim text block dominate Euclidean
        distance purely by dimension count, drowning the 10 numeric features. Both
        blocks are standardized to ~unit variance per dimension upstream (text by
        self.text_scaler_, numeric by self.scaler_), so I rescale each block by
        1/sqrt(#dims): both then contribute equal *total* variance. The two weights
        let me deliberately tilt toward topic or writing-style.
        """
        text = text_svd / np.sqrt(text_svd.shape[1]) * self.text_weight
        num = num_scaled / np.sqrt(num_scaled.shape[1]) * self.numeric_weight
        return np.hstack([text, num]).astype("float32")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self._fit_activity(df)
        text_svd = self._fit_text(df)
        # standardize both blocks per-dimension before balancing
        self.text_scaler_ = StandardScaler().fit(text_svd)
        num_prepped = self._prep_numeric(df)
        self.scaler_ = StandardScaler().fit(num_prepped)
        return self._combine(
            self.text_scaler_.transform(text_svd),
            self.scaler_.transform(num_prepped),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        text_svd = self._transform_text(df)
        num_prepped = self._prep_numeric(df)
        return self._combine(
            self.text_scaler_.transform(text_svd),
            self.scaler_.transform(num_prepped),
        )
