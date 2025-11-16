"""Module to cluster and visualise ECLASS embeddings using DBSCAN and UMAP."""

import logging
import textwrap
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from src.utils.json_io import load_json
from src.utils.logger import LoggerFactory


def wrap_text(s: str, width: int) -> str:
    """Wraps a string to the given width with <br> line breaks."""

    if not isinstance(s, str):
        return ""
    return "<br>".join(textwrap.wrap(s, width=width, break_long_words=True, replace_whitespace=False))


def first_n_words(s: str, n: int) -> str:
    """Returns the first n words from a string."""

    if not isinstance(s, str):
        return ""
    words = s.split()
    return " ".join(words[:n])


def cluster_and_plot(
        input_path: str,
        output_path: str,
        model: str,
        segment: str,
        logger: logging.Logger
) -> None:
    """Loads embeddings, runs DBSCAN clustering, reduces with UMAP, and saves a Plotly HTML scatter."""

    logger.info(f"Starting clustering ...")

    # Load embeddings from JSON file
    try:
        database = load_json(input_path)
        logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        return
    df = pd.DataFrame(database)
    embeddings = np.array(df["embedding"].tolist())

    # Check that enough data points for clustering are available
    if len(embeddings) < 5:
        logger.warning(f"Segment {segment} has too few points ({len(embeddings)}), skipping clustering and UMAP.")
        return

    # DBSCAN clustering
    logger.info("Computing clusters ...")
    embeddings = normalize(embeddings, norm='l2')
    dbscan = DBSCAN(eps=0.2, min_samples=3, metric="cosine")
    labels = dbscan.fit_predict(embeddings)
    df["cluster"] = labels

    unique_clusters = sorted(set(labels))
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise
        examples = df[df["cluster"] == cluster]["preferred-name"].head(5).tolist()
        logger.info(f"Segment {segment} cluster {cluster} examples: {examples}")

    # UMAP reduction to 2D
    reducer = umap.UMAP(metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Pre-wrap text columns
    df["preferred-name-wrapped"] = df["preferred-name"].apply(lambda s: wrap_text(first_n_words(s, 3), width=16))
    df["definition"] = df["definition"].apply(lambda s: wrap_text(s, width=115))

    # Plot
    signal_palette = [
        "#0072B2",  # Blue
        "#D55E00",  # Red
        "#009E73",  # Green
        "#E69F00",  # Orange
        "#CC79A7",  # Purple
        "#56B4E9",  # Sky blue
        "#F0E442",  # Yellow
    ]

    label_strs = [str(c) for c in unique_clusters]
    color_discrete_map = {"-1": "#000000"}  # Noise = black

    remaining_labels = [s for s in label_strs if s != "-1"]
    palette_iter = cycle(signal_palette)
    for lbl in remaining_labels:
        color_discrete_map[lbl] = next(palette_iter)

    show_labels = len(df) <= 300  # Only render labels when few points

    df["cluster"] = df["cluster"].astype(str)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data={
            "id": True,
            "preferred-name": True,
            "preferred-name-wrapped": False,
            "definition": True,
            "x": False,
            "y": False,
            "cluster": False
        },
        title=f"ECLASS Clustering - Embedding Model {model.capitalize()} - Segment {segment}",
        category_orders={"cluster": ["-1"] + remaining_labels},
        color_discrete_map=color_discrete_map,
        text="preferred-name-wrapped" if show_labels else None  # Small name above point
    )

    fig.update_traces(
        mode="markers+text",
        marker=dict(size=5, opacity=0.75),
        textposition="top center",
        textfont=dict(size=8),
        selector=dict(mode='markers+text')
    )

    # Match text colour to point colour per trace
    fig.for_each_trace(lambda tr: tr.update(textfont_color=tr.marker.color))

    fig.update_layout(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        showlegend=True,
    )

    # Save HTML
    fig.write_html(output_path)
    logger.info(f"DBSCAN plot saved to: {output_path}")


if __name__ == "__main__":
    # Settings
    run_per_segment = False  # Compute clusters per segment or in one run
    exceptions = []  # Exclude specific segments if run per segment
    transformer = "bge"  # Cluster embeddings from this transformer (qwen3, bge, gemini)

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    logger.info(f"Initialising clustering ...")

    if run_per_segment:
        # Compute clusters for each segment
        segments = list(range(13, 52)) + [90]
        for segment in segments:
            if segment in exceptions:
                logger.warning(f"Skipping segment {segment}.")
                continue
            input_path = f"../../data/embedded-classes/eclass-{segment}-embeddings-{transformer}.json"
            output_path = f"../../visualisation/eclass-{segment}-embeddings-{transformer}.html"
            cluster_and_plot(input_path, output_path, transformer, str(segment), logger)
    else:
        # Compute clusters for combined segments
        input_path = f"../../data/embedded-classes/eclass-0-embeddings-{transformer}.json"
        output_path = f"../../visualisation/eclass-0-embeddings-{transformer}.html"
        cluster_and_plot(input_path, output_path, transformer, "All", logger)
