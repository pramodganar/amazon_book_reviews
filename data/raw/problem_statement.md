# Problem statement — Amazon Book Reviews Clustering

Restated from the original assignment, with the dataset facts corrected to match
the data actually delivered (see the dataset notes in the README and
`reports/report.md` section 1 for the discrepancy).

## Objective

Apply unsupervised clustering to group book reviews by their content, writing
style, and reviewer behavior. There is no labeled target (no classification or
regression); the task uncovers natural groupings using methods such as K-Means,
DBSCAN, or hierarchical clustering.

## Goals

- Discover latent reviewer/customer segments (e.g., harsh critics, enthusiastic
  fans, academic reviewers).
- Identify common review patterns or topics (e.g., praise for illustrations,
  academic analysis).
- Support targeted marketing, product improvement, or fake-review detection by
  analyzing clusters of similar reviews.

## Data

Use the `amazon_book_reviews` table from the clustering SQLite database
(`clustering.db`; see `Guidelines_to_fetch_data_from_Database.docx`).

The delivered table has **300,000 records** and six columns:
`Id, Title, User_id, profileName, review/time, review/text`. The original brief
described ~3 million reviews and additional rating / helpfulness / price / summary
fields; none of those columns exist in the actual data, so there is no rating or
helpfulness signal to cluster on.

## Split

Time-ordered split into model-building / evaluation / live records. The original
brief specified 2,100,000 / 600,000 / remainder — a 70 / 20 / 10 ratio scaled to
the overstated 3M. This project applies the same **70 / 20 / 10** split to the
298,824 cleaned rows (~209k / 60k / 30k).

## Deliverables

- Separate `.py` files for model training and prediction.
- Identical data-processing and feature-creation steps in the build and prediction
  phases.
- A final report covering: model selected, features selected, feature importance,
  model evaluation, and the original code.
