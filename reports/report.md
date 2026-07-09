# Amazon Book Reviews: Unsupervised Reviewer Segmentation

Grouping ~300k Amazon book reviews into natural reviewer segments and surfacing
review topics, with no labels. This report covers what the data actually is, the
features and model I chose, how I read the clusters, how they hold up out of
sample, and (importantly) where the approach breaks down.

## 1. Data

Source: the `amazon_book_reviews` table in the program's SQLite `clustering.db`
(per `data/raw/Guidelines_to_fetch_data_from_Database.docx`), with a CSV export
that is what's actually on disk here. The accompanying Data Card advertises ten columns
including `Price`, `review/score`, `review/helpfulness`, and `review/summary`.
**None of those exist in the data.** The real schema is six columns:

`Id, Title, User_id, profileName, review/time, review/text`

So there is no star rating and no helpfulness signal to lean on, and every feature
is derived from the text, its shape, reviewer activity, and time. Row count is
**300,000** (the problem doc's "3M" is wrong; confirmed in code). Two analyses you
might expect are therefore N/A here by construction: rating-distribution skew, and
any check for rating leaking into the text features - there is no rating column to
plot or to leak. That also removes a common clustering pitfall for free.

Cleaning (all row-level and stateless, so predict-time data gets identical rules):

| Issue | Count | Action |
|---|---|---|
| Exact duplicate reviews | 1,163 | drop (they form artificial dense clusters) |
| Sentinel timestamps (unix <= 0, decoding to 1969) | 7 | drop |
| Empty / under-3-char text | ~6 | drop (no signal) |
| Null `User_id` | 57,999 (19.3%) | **keep** as distinct anonymous reviewers |

Result: 300,000 down to **298,824** rows. Null users are kept rather than collapsed
into one placeholder, which would have fabricated a single 58k-review "user".

## 2. Split

Time-ordered 70/20/10 by `review/time`, to mimic production (fit on the past,
score the future):

- train 209,176 (1995 to 2009-03)
- eval  59,765 (2009-03 to 2012-09)
- live  29,883 (2012-09 to 2013-03)

Everything stateful (TF-IDF, SVD, scalers, cluster model, activity counts) is fit
on **train only**; eval and live are transform-only. Verified there is no time
leakage across boundaries.

## 3. Features (210 dimensions)

Two blocks, kept separable so each cluster can be inspected on both:

**Text (200 dims).** Clean text (lowercase, unescape HTML entities, strip tags,
URLs and non-letters), then TF-IDF (20k vocab cap, `min_df=5`, `max_df=0.5`,
`sublinear_tf`, English plus custom stopwords), then TruncatedSVD to 200 dims (LSA).
Raw TF-IDF is ~20k-dim and sparse, useless for distance-based clustering, and SVD
makes it tractable. The custom stopwords remove HTML-entity remnants (`quot`,
`amp`, `lt`) and contraction fragments (`ve`, `ll`) that otherwise polluted the top
topics.

**Numeric (10 dims).**
- *Structural / tone* (from raw text): char length, word count, avg word length,
  uppercase ratio, exclamation count, question count.
- *Temporal*: day-of-week, year. (`hour` was dropped because every timestamp is
  midnight-aligned, so it carried a single value.)
- *Activity*: reviews-per-user and reviews-per-product, learned from train;
  unseen and anonymous reviewers map to 1.

The six power-law numeric features are `log1p`-transformed and `avg_word_len` is
clipped before standardizing, so a handful of 30k-char reviews or 500-review users
don't dominate Euclidean distance.

**Block balancing.** Concatenating 200 text dims with 10 numeric dims would let
text dominate distance purely by dimension count. Each block is standardized
per-dimension, then rescaled by `1/sqrt(#dims)` so both contribute equal total
variance. Two tunable weights allow tilting toward topic vs. writing style. That
this mattered is confirmed below: `user_review_count` and `dow` end up among the
strongest cluster separators, which is only possible because numeric wasn't
drowned. A weight sweep (`experiments/ablations.py`) backs the equal default:
doubling the text weight collapses the structure (silhouette 0.001, one 34%
mega-cluster), doubling the numeric weight lifts silhouette to 0.09 but
degenerates the shares (a 0.5% fragment); equal weights keep the 8–21% balance.
The equal-weight path also reproduces the shipped labels exactly (ARI 1.0), a
parity check on the pipeline.

## 4. Model selection

Compared three algorithms on the 210-dim train matrix.
`experiments/model_selection.py` reproduces every number in this section and
regenerates `reports/figures/k_selection.png`.

**MiniBatchKMeans (chosen).** Scales to 210k, fast, gives balanced partitions
after SVD. Swept k from 2 to 15 on inertia (elbow), silhouette, Davies-Bouldin and
Calinski-Harabasz. No metric singles out a k: silhouette is highest at the
degenerate k=2 (0.064), then sits around 0.03 to 0.04 for every larger k (a slight
nudge to 0.040 at k=8), and inertia falls smoothly with no sharp elbow. That the
metrics don't pick a k is itself the finding.

**DBSCAN (rejected).** On a 20k sample, no `eps` yields balanced clusters. Below
`eps` ~1.0 almost everything is noise (96 to 100%) with only tiny fragments; by
`eps` 1.5 to 2.0 it collapses into a single cluster holding 95 to 99% of points.
The 5th-NN distances are bunched (0.9 to 1.5, 10th to 90th percentile), so no
radius separates dense from sparse. There is no density structure in 210-dim space
(curse of dimensionality). A clean negative result, not a misconfiguration.

**Agglomerative (sanity check only).** O(n^2) memory, so subsample-only (5k).
Silhouette is weakly positive throughout and fades with k (0.051 at k=2 to ~0.012
by k=8 to 10 - never negative, but never strong), and the partitions stay lopsided
(a ~25-point micro-cluster persists from k=4 on). Same continuum-like structure as
KMeans, no clean partition at any k.

**Hyperparameter sensitivity (`experiments/sensitivity.py`).** Sweeping SVD
dimensionality on the shipped TF-IDF: 200 components is the silhouette peak
(0.035, vs 0.026 at 50, 0.031 at 100, 0.029 at 400) — d=400 retains 24% of the
variance but clusters worse and produces a near-empty cluster, so extra dimensions
add noise faster than signal. Varying the TF-IDF knobs one at a time (min_df 2/20,
max_df 0.3/0.8, vocabulary 10k/40k) moves the exact partition roughly as much as
changing the seed does (ARI 0.20–0.27 against the shipped labels; the seed baseline
is ~0.32) — under weak separation, per-review labels churn with any perturbation —
but no variant uncovers materially stronger balanced structure: the settings that
lift silhouette (10k vocab, 0.060; max_df=0.3, 0.042) do it while carving out
0.1–0.5% fragment clusters, whereas the shipped configuration keeps every share
between 8 and 21%.

**The headline result: these reviews are a continuum, not crisp clusters.**
Silhouette never exceeds ~0.064 (at the degenerate k=2) and stays around 0.03 to
0.04 for every usable k; inertia has no sharp elbow. This is expected for
high-dimensional text and I'm not going to pretend otherwise. k=2 is
metric-"optimal" but degenerate for segmentation. I chose **k = 6**: the most
balanced, interpretable partition (six segments of 8 to 21% each, section 5) with
no metric contradicting it. Higher k nudges silhouette by ~0.005 - within the noise
of a sub-0.05 range - while shaving off small fragments (Agglomerative already
isolates a 25-point cluster) that don't earn a persona.

## 5. The six segments and their "importance"

With no target, "feature importance" is three complementary views: top distinctive
TF-IDF terms per cluster, ANOVA F-stat of each numeric feature across labels, and
centroid profiles.

| # | n | Label | Defining signal |
|---|---|---|---|
| 0 | 40k | **How-to / reference reviewers** | *helpful, useful, guide, illustrations*; short (317 ch) |
| 1 | 44k | **Punchy popular-book reactors** | *loved, best, wait, buy*; short, high exclaim, popular titles |
| 2 | 17k | **Prolific plot-summarizers** | *novel, tale, war*; long (1694 ch), heavy reviewers (55/user) |
| 3 | 42k | **General opinion reviewers** | *think, like, thought*; medium length; softest/catch-all |
| 4 | 22k | **Argumentative deep-divers** | *does, question, people*; long, question_count 2.2 |
| 5 | 44k | **Analytical critics / essayists** | *reader, author, history, fact*; long, zero questions |

**ANOVA F (numeric features that separate clusters most):** `question_count`
(39,199) far ahead of `char_len`/`word_count` (~27,000), then `dow` (19,969),
`user_review_count` (10,521), `product_review_count` (4,851). So the segmentation
is driven by *tone* (questioning/rhetorical), *length* (short reaction vs long
essay), and *behavior* (prolific vs one-off), not topic alone.

## 6. Evaluation and stability

Eval/live scored with the fitted centroids (no refit). `evaluate.py` reproduces
this table and the distance drift below:

| split | silhouette | davies-bouldin | calinski-h |
|---|---|---|---|
| train | 0.034 | 3.45 | 8973 |
| eval  | 0.051 | 3.33 | 2671 |
| live  | 0.023 | 3.45 | 763 |

Geometry holds across splits; clusters don't collapse out of sample. Eval scoring
slightly above train (0.051 vs 0.034) is mostly era homogeneity, not a fluke: on
identical features and labels, a same-size window of late train scores 0.045
against 0.036 for the full 1995–2009 era (`experiments/ablations.py`).

**Seed stability (Adjusted Rand Index).** Refitting MiniBatchKMeans at five seeds
on the *same* feature matrix (only the clustering init changes) gives a pairwise
ARI of **mean 0.32, min 0.24, max 0.39** (`experiments/seed_stability.py`). That's
low: the exact point-level partition is materially init-dependent. It's the direct
consequence of the weak separation in section 4, not a bug - with no real gaps
between groups, boundary reviews reassign freely between seeds. What is stable is
the *coarse* structure: every seed produces a short-reaction group, a long-essay
group, and a questioning group with recognisably similar centroids and top terms;
it's the soft catch-all (c3) and its neighbours that churn. So the personas are
reproducible tendencies; the precise membership is not. In production I'd fix the
seed for a deterministic model and treat the segment definitions, not any single
review's label, as the deliverable.

**Is the churn the data or the mini-batch?** Partly the algorithm. Full KMeans on
the same matrix (`experiments/ablations.py`) is materially more seed-stable
(pairwise ARI mean 0.58 vs 0.32; two of three seeds near-identical at 0.99) and
reaches silhouette 0.056–0.060 against the shipped 0.034, at 30–60s per fit. The
continuum conclusion stands — 0.06 is still weak separation — but with hindsight
full KMeans is the better trade at this scale, and refitting with it is the first
improvement listed in section 8.

**Cluster-share drift over time** is the most interesting result:

- **c2 (prolific reviewers) nearly disappears on live: 8% to 3.6% to 0.5%.** This is
  a direct consequence of a feature-design choice, not a bug. `user_review_count`
  is learned from train; future reviewers are almost all unseen and map to 1, and
  that cluster is *defined* by high user activity, so almost nothing new can land
  in it. The activity-feature limitation (section 3) shows up as measured drift.
- **c0 (short how-to reviews) grows to 50% on live**, because reviews genuinely got
  shorter from 2009 to 2013.

**Concept drift:** every cluster's mean distance-to-centroid grows on live
(+0.25 to +0.51), and the live distance distribution has a long tail (median 1.6,
max 6.9). In production this model would need periodic refitting. That far tail is
also a natural **anomaly / fake-review detection** hook: reviews sitting far from
every centroid are the outliers to inspect.

## 7. Limitations

1. **Weak separation.** Low silhouette everywhere; the segments are real tendencies
   along a continuum, not clean partitions. c3 in particular is a soft catch-all.
2. **Activity features don't transfer to the future.** Because most future
   reviewers are new, `user_review_count` collapses to 1 out of sample, killing the
   prolific-reviewer cluster at predict time. A cumulative or externally-sourced
   user history would fix this; the current data can't.
3. **`dow` ranks surprisingly high** as a separator. That's more an artifact of a
   low-cardinality axis KMeans latched onto than a real "weekend reviewer" persona,
   so I don't over-read it. Measured (`experiments/ablations.py`): refitting
   without `dow` moves the partition no more than a seed change does (ARI 0.31
   against the ~0.32 seed-churn baseline), and the distinctive personas persist —
   the questioning cluster maps 96% onto its shipped counterpart, the prolific one
   91%. `dow` inflates the ANOVA table but is not load-bearing.
4. **LSA retains ~16% of TF-IDF variance at 200 components.** Normal for short,
   lexically diverse text, but it means topic structure is diffuse. The dimension
   sweep (section 4) shows 200 is nonetheless the right operating point: 400
   components double the retained variance yet cluster worse.
5. **Near-duplicate reviews** (same text and product, different user; ~550) were
   kept for safety; only exact duplicates were dropped.
6. **Seed sensitivity.** Pairwise ARI across seeds is only ~0.32 (section 6): the
   partition is reproducible at the persona level but not at the individual-label
   level. A downstream use that needs a hard, stable per-review segment would want
   fewer clusters or a soft-assignment model; k=6 is chosen for interpretability,
   with the boundary churn disclosed rather than hidden.

## 8. Possible next steps

Refit the shipped model with full KMeans — measurably more seed-stable and
better-scoring than MiniBatch at this scale (section 6) for ~a minute of fit time.
Sentence-transformer embeddings in place of TF-IDF+SVD (denser, better topic
separation); LDA or BERTopic for explicit per-cluster topic labels; a cumulative
user-history feature to make activity transfer across time; and turning the
distance-to-centroid tail into an explicit anomaly score for fake-review triage.
