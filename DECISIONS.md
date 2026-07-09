# Design decisions

One line each: what was decided and why, with a pointer to where the code or report
backs it. Items that would need an experiment the repo doesn't have are marked
"open gap" and cross-referenced in [INTERVIEW_QA.md](INTERVIEW_QA.md).

## Data

- **Keep 19% null `User_id` as distinct anonymous reviewers, not one placeholder
  user.** A shared placeholder would fabricate a single 58k-review "user" and poison
  the activity features (`pipeline.py:141-144`).
- **Drop only exact duplicates; keep ~550 near-duplicates.** Same text under a
  different user could be a legitimate cross-post; with no ground truth, dropping
  them risks deleting real reviews, and 550 of 300k cannot move a centroid
  (`reports/report.md` §7.5).
- **Drop `review/time <= 0`.** Sentinel values decode to 1969 and would pile up at
  the front of a time-ordered train split (`pipeline.py:131-135`).
- **Correct the brief's "3M rows / 10 columns" to the delivered 300k / 6 columns in
  a rewritten problem statement.** Confirmed by loading the delivered table directly;
  the original docx remains in git history (removed in `6767c78`), so the corrected
  `problem_statement.md` and the evidence for the correction are both preserved.

## Split

- **Time-ordered 70/20/10 instead of random, despite no target.** Activity counts
  are learned from data, so a random split would let a reviewer's future activity
  leak into their count; it also makes drift measurable (`pipeline.py:47-50`).

## Features

- **TF-IDF → TruncatedSVD(200).** Raw TF-IDF is ~20k-dim and sparse — unusable for
  Euclidean KMeans; 200 is a conventional LSA operating point that keeps the fit
  cheap, and the retained variance (~16%) is reported rather than hidden
  (`pipeline.py:65-68`). The sweep (`experiments/sensitivity.py`) backs it: 200 is
  the silhouette peak (0.035 vs 0.026–0.031 at 50/100/400), and 400 doubles the
  retained variance but clusters worse with a near-empty cluster.
- **`min_df=5`, `max_df=0.5`, 20k vocabulary cap.** Conventional guards, not tuned:
  `min_df` drops typos and IDs, `max_df` drops terms too common to separate anyone,
  the cap bounds memory (`pipeline.py:58-63`). Measured (`experiments/sensitivity.py`):
  one-knob variants move exact labels about as much as a seed change (ARI 0.20–0.27,
  expected under weak separation) and no variant finds stronger balanced structure —
  the ones that lift silhouette create 0.1–0.5% fragment clusters.
- **Custom stopwords for contraction/entity fragments.** They survive `min_df` and
  polluted top terms (`pipeline.py:229-234`).
- **Structural features computed on RAW text before any cleaning.** Uppercase ratio
  and exclamation counts only exist in the original casing/punctuation
  (`pipeline.py:190-195`).
- **`log1p` on the six power-law features; clip `avg_word_len` at 15.** Prevents a
  few 30k-char reviews or 500-review users from dominating Euclidean distance
  (`pipeline.py:70-78`).
- **Drop `hour`; keep `dow` and `year`.** Every timestamp is midnight-aligned, so
  hour is constant (`pipeline.py:184`). `dow` is kept because it is cheap and
  interpretable, and its suspiciously high ANOVA rank is disclosed as a
  low-cardinality artifact rather than read as a persona (`reports/report.md` §7.3).
  Measured: refitting without it moves the partition no more than a seed change
  (ARI 0.31 vs the ~0.32 seed baseline) and the distinctive personas persist
  (`experiments/ablations.py`).
- **Block balancing: standardize both blocks, then scale each by 1/sqrt(#dims).**
  Otherwise 200 text dims dominate 10 numeric dims by count alone; equal total
  variance is the neutral default absent a reason to prefer topic over style
  (`pipeline.py:347-359`). A sweep of the `text_weight`/`numeric_weight` knobs
  confirms it (`experiments/ablations.py`): doubling text collapses structure,
  doubling numeric degenerates the cluster shares; equal weights win.

## Model

- **MiniBatchKMeans over full KMeans.** The k=2..15 sweep plus five-seed stability
  runs mean dozens of fits at `n_init=10`; MiniBatch keeps each one cheap at 209k
  rows (`train.py:45-48`). The side-by-side (`experiments/ablations.py`) shows the
  cost of that choice: full KMeans is more seed-stable (pairwise ARI 0.58 vs 0.32)
  and scores higher (silhouette ~0.06 vs 0.034) at 30–60s per fit, so refitting
  with full KMeans is the first listed improvement (`reports/report.md` §8) — the
  continuum conclusion is unchanged either way.
- **k=6 despite no metric picking it.** Silhouette is metric-optimal at the
  degenerate k=2 and flat (~0.03–0.04) for k≥3; six is the most balanced,
  interpretable partition and no metric contradicts it (`reports/report.md` §4).
- **DBSCAN rejected with evidence, not opinion.** 5th-NN distances are bunched, so
  no eps separates dense from sparse in 210-dim space
  (`experiments/model_selection.py:69-87`).
- **Fixed seed (42) shipped; instability measured and disclosed.** Pairwise ARI
  across seeds ~0.32; the deliverable is the segment definitions, not per-review
  label permanence (`experiments/seed_stability.py`, `reports/report.md` §6).

## Packaging / app

- **Ship the fitted artifacts (35MB joblib) in git.** The hosted Streamlit demo
  builds from the repo and cannot run a 260MB-data training step, so committing the
  fitted pipeline is the simplest deploy that keeps the demo live (commit
  `f28e7ef`). Git LFS was avoided because Streamlit Cloud's LFS support is
  unreliable; a release asset plus a download step is the cleaner alternative if
  the artifacts grow.
- **App classification uses a fixed `review/time` (2012-10, the live-era midpoint)
  and a synthetic Id/User.** A pasted review has no timestamp or product; the
  choice and its effects (temporal features constant, activity counts default to 1)
  are commented in `app.py` and disclosed in the app caption.
- **Tests use tiny synthetic frames, not the dataset.** CI and a fresh clone can
  run the suite with no data (`tests/test_pipeline.py:1-5`).
