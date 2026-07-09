"""
Streamlit front-end for the book review segmentation model.

    streamlit run app.py

Loads the artifacts fit by train.py and lets you classify a review, browse the
six segments, and see how the model was chosen and how it holds up out of sample.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as P

st.set_page_config(page_title="Book Review Segmenter", layout="wide")

ARTIFACTS = P.ARTIFACT_DIR


@st.cache_resource
def load_model():
    fp = joblib.load(ARTIFACTS / "feature_pipeline.joblib")
    km = joblib.load(ARTIFACTS / "kmeans.joblib")
    with open(ARTIFACTS / "cluster_meta.json") as f:
        meta = json.load(f)
    with open(ARTIFACTS / "config.json") as f:
        config = json.load(f)
    return fp, km, meta, config


if not (ARTIFACTS / "feature_pipeline.joblib").exists():
    st.error("No trained artifacts found. Run `python train.py` and "
             "`python analyze_clusters.py` first.")
    st.stop()

fp, km, meta, config = load_model()
LABELS = {int(k): v for k, v in meta["labels"].items()}
COLORS = px.colors.qualitative.Set2


def label_of(c: int) -> str:
    return f"c{c}: {LABELS[c]}"


st.title("Amazon Book Review Segmenter")
st.caption("Unsupervised clustering of ~300k reviews into reviewer segments. "
           "No labels; built from text, writing style, activity and time.")

tab_try, tab_segments, tab_map, tab_model = st.tabs(
    ["Classify a review", "The 6 segments", "Cluster map", "Model & evaluation"])


# classify a single review
with tab_try:
    st.subheader("Paste a review and see its segment")
    examples = {
        "(pick an example)": "",
        "Enthusiastic fan": "Absolutely loved this book!! Best read of the year, "
                            "could not put it down. Buy it now!!!",
        "How-to / reference": "A clear, helpful guide. The illustrations and step-by-step "
                             "diagrams make it easy to follow. Great reference for beginners.",
        "Analytical essay": "The author marshals extensive archival research to reconstruct "
                           "the campaign, situating the narrative within the broader "
                           "historiography of the period. The argument is measured and "
                           "grounded in primary sources.",
        "Argumentative": "Is this really the best the author could do? Why so much filler? "
                        "Who edited this? Does the premise even hold up? What were they thinking?",
    }
    pick = st.selectbox("Or start from an example", list(examples))
    text = st.text_area("Review text", value=examples[pick], height=160,
                        placeholder="Type or paste a book review here...")

    if st.button("Classify", type="primary") and text.strip():
        # A pasted review has no real timestamp or product: pin the time to the
        # live-era midpoint (2012-10) so temporal features are era-consistent,
        # and use a synthetic Id/User so activity counts take the unseen default.
        # Both choices are disclosed in the caption below.
        row = pd.DataFrame({
            "Id": ["APP"], "Title": ["app"], "User_id": [None], "profileName": [None],
            "review/time": [1350000000], "review/text": [text],
        })
        cleaned = P.clean(row)
        if cleaned.empty:
            st.warning("Review too short to classify: the pipeline drops reviews "
                       f"under {P.MIN_TEXT_CHARS} characters (no usable signal).")
        else:
            X = fp.transform(cleaned)
            c = int(km.predict(X)[0])
            dists = km.transform(X)[0]

            st.markdown(f"### {LABELS[c]}  (cluster {c})")

            col1, col2 = st.columns([3, 2])
            with col1:
                # shorter bar = closer centroid = better fit
                aff = pd.DataFrame({
                    "segment": [label_of(i) for i in range(len(dists))],
                    "distance": dists,
                }).sort_values("distance")
                fig = px.bar(aff, x="distance", y="segment", orientation="h",
                            title="Distance to each segment centroid (shorter is a better fit)",
                            color="segment", color_discrete_sequence=COLORS)
                fig.update_layout(showlegend=False, height=300, yaxis_title="")
                st.plotly_chart(fig, width="stretch")
            with col2:
                st.caption("What the model saw in this text")
                feats = fp.transform_numeric(cleaned).iloc[0]
                st.dataframe(feats.rename("value").to_frame().style.format("{:.2f}"),
                            width="stretch")

            st.info("Top terms in this segment: "
                    + ", ".join(meta["clusters"][str(c)]["top_terms"][:10]))
            st.caption("An unseen product or user defaults to an activity count of 1, so a "
                       "short review of an obscure title tends toward the short-review "
                       "segments. Pasted text also has no real timestamp; it is pinned to "
                       "late 2012 (the live era), so temporal features are constant here.")


# the six segments
with tab_segments:
    st.subheader("The six reviewer segments")
    shares = pd.DataFrame({
        "segment": [label_of(c) for c in range(P.N_CLUSTERS)],
        "share": [meta["clusters"][str(c)]["share"] for c in range(P.N_CLUSTERS)],
        "size": [meta["clusters"][str(c)]["size"] for c in range(P.N_CLUSTERS)],
    })
    fig = px.bar(shares, x="share", y="segment", orientation="h", text="size",
                color="segment", color_discrete_sequence=COLORS,
                title="Train-split share of each segment")
    fig.update_layout(showlegend=False, height=320, xaxis_tickformat=".0%", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    for c in range(P.N_CLUSTERS):
        cm = meta["clusters"][str(c)]
        with st.expander(f"{label_of(c)}  ({cm['size']:,} reviews, {cm['share']:.1%})"):
            st.write("Top distinctive terms: " + ", ".join(cm["top_terms"]))
            prof = cm["profile"]
            st.write(
                f"Profile: about {prof['char_len']:.0f} chars, "
                f"{prof['word_count']:.0f} words, "
                f"{prof['question_count']:.2f} questions, "
                f"{prof['exclaim_count']:.2f} exclamations, "
                f"{prof['user_review_count']:.1f} reviews/user, "
                f"{prof['product_review_count']:.0f} reviews/product."
            )


# 2d cluster map
with tab_map:
    st.subheader("2D map of the segments")
    scatter_path = ARTIFACTS / "scatter.parquet"
    if not scatter_path.exists():
        st.warning("Run `python analyze_clusters.py` to build the cluster map.")
    else:
        sc = pd.read_parquet(scatter_path)
        sc["segment"] = sc["cluster"].map(lambda c: label_of(int(c)))
        fig = px.scatter(
            sc.sort_values("cluster"), x="x", y="y", color="segment",
            hover_data={"snippet": True, "x": False, "y": False, "segment": False},
            color_discrete_sequence=COLORS, opacity=0.7,
            title=f"t-SNE projection of a {len(sc):,}-review sample",
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=560, legend_title="", xaxis_title="", yaxis_title="",
                          xaxis_showticklabels=False, yaxis_showticklabels=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("t-SNE is only for visualization; distances between points are not "
                   "meaningful. The segments overlap rather than sitting in tidy islands: "
                   "the reviews form a continuum. Hover to read each sampled review.")


# model selection and evaluation
with tab_model:
    st.subheader("How the model was chosen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews (train)", f"{config['n_train']:,}")
    c2.metric("Feature dimensions", config["feature_dim"])
    c3.metric("Segments (k)", config["n_clusters"])

    st.markdown(
        "- MiniBatchKMeans chosen: scales to 210k, gives balanced segments after SVD.\n"
        "- DBSCAN rejected: collapses to one cluster plus noise in 210-dim space.\n"
        "- Silhouette is low across all k (the reviews are a continuum, not crisp "
        "clusters). k=2 is metric-optimal but degenerate, so k=6 for interpretability."
    )
    fig_path = P.PROJECT_DIR / "reports" / "figures" / "k_selection.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="k-selection: elbow and silhouette/Davies-Bouldin")

    st.markdown("#### Feature importance (ANOVA F across clusters)")
    anova = pd.DataFrame({"feature": list(meta["anova"]), "F": list(meta["anova"].values())})
    fig = px.bar(anova, x="F", y="feature", orientation="h",
                color_discrete_sequence=["#4C78A8"])
    fig.update_layout(height=320, yaxis_title="", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    tm = config["train_metrics"]
    st.markdown(
        f"Train internal metrics: silhouette {tm['silhouette']:.3f}, "
        f"Davies-Bouldin {tm['davies_bouldin']:.2f}, "
        f"Calinski-Harabasz {tm['calinski_harabasz']:.0f}."
    )

    if "drift" in meta:
        st.markdown("#### Segment drift over time (train, eval, live)")
        drift = meta["drift"]
        rows = []
        for split in ("train", "eval", "live"):
            for c in range(P.N_CLUSTERS):
                rows.append({"split": split, "segment": label_of(c),
                             "share": drift[split][str(c)]})
        dd = pd.DataFrame(rows)
        fig = px.bar(dd, x="segment", y="share", color="split", barmode="group",
                     category_orders={"split": ["train", "eval", "live"]},
                     color_discrete_sequence=["#4C78A8", "#F58518", "#54A24B"])
        fig.update_layout(height=380, yaxis_tickformat=".0%", xaxis_title="",
                          legend_title="", xaxis_tickangle=-20)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"The prolific plot-summarizer segment collapses from "
            f"{drift['train']['2']:.1%} on train to {drift['live']['2']:.1%} on live: "
            "activity features do not transfer to unseen future reviewers, so almost "
            "nothing new can land there. The short how-to segment balloons as reviews "
            "got shorter over 2009 to 2013. This is real temporal drift, and in "
            "production the model would need periodic refitting."
        )

    st.caption("Limitation to keep in mind: the segments are tendencies along a "
               "continuum, not sharp partitions (see reports/report.md).")
