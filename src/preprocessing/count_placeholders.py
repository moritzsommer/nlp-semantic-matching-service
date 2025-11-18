"""Module to visualise non-semantic ECLASS definitions using predefined filters."""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.filter_non_semantic import (
    filter_definitions_missing, filter_definitions_missing_suffix,
    filter_definitions_structural)
from src.utils.logger import LoggerFactory


def plot_nonsemantic_ratios(
        input_path: str,
        segments: List[int],
        exceptions: List[int],
        output_path: str,
        logger: logging.Logger,
) -> None:
    """Plots the ratio of non-semantic definitions per ECLASS segment."""

    logger.info("Starting non-semantic definition ratio plot ...")

    # Prepare fast lookups
    missing_set = set(filter_definitions_missing)
    suffixes = tuple(filter_definitions_missing_suffix)
    structural_set = set(filter_definitions_structural)

    # Containers
    total_per_seg: Dict[int, int] = {}
    miss_per_seg: Dict[int, int] = {}
    struct_per_seg: Dict[int, int] = {}
    processed_segments: List[int] = []

    # Run for each segment
    for seg in segments:
        if seg in exceptions:
            logger.warning(f"Skipping segment {seg}.")
            continue

        # Load ECLASS classes from CSV file
        try:
            database = pd.read_csv(input_path.format(segment=seg), sep=",")
            logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
        except Exception as e:
            logger.error(f"Failed to read file: {input_path}, Error: {e}")
            continue

        # Ensure required columns exist
        required_columns = ["definition", "preferred-name", "id"]
        for col in required_columns:
            if col not in database.columns:
                logger.error(f"Missing required column: {col}")
                continue

        # Count non-semantic definitions
        total = len(database)
        total_per_seg[seg] = total
        processed_segments.append(seg)

        n_missing = 0
        n_structural = 0

        for definition in database["definition"].dropna().astype(str):
            if (definition in missing_set) or definition.endswith(suffixes):
                n_missing += 1
            elif definition in structural_set:
                n_structural += 1

        miss_per_seg[seg] = n_missing
        struct_per_seg[seg] = n_structural

        logger.info(f"Segment {seg}: total={total}, missing={n_missing}, structural={n_structural}")

    if not processed_segments:
        logger.error("No segments processed, nothing to plot.")
        return

    # Build percentage arrays aligned with processed segment order
    x_labels = [str(s) for s in processed_segments]
    miss_pct = [
        (miss_per_seg.get(s, 0) / max(total_per_seg.get(s, 0), 1)) * 100.0 for s in processed_segments
    ]
    struct_pct = [
        (struct_per_seg.get(s, 0) / max(total_per_seg.get(s, 0), 1)) * 100.0 for s in processed_segments
    ]

    # Plot
    plt.figure(figsize=(20, 8), dpi=300)

    color_missing = "#4E5F72"  # Dark blue, adapted from standard #7D91A6 with red +9, green +11, blue + 13
    color_struct = "#7D91A6"  # Light blue from standard colour palette

    bottom = [0.0] * len(processed_segments)
    plt.bar(
        x_labels,
        miss_pct,
        bottom=bottom,
        label="Missing definition",
        color=color_missing,
        edgecolor="black",
        linewidth=0.3
    )
    bottom = [b + h for b, h in zip(bottom, miss_pct)]
    plt.bar(
        x_labels,
        struct_pct,
        bottom=bottom,
        label="Structural definition",
        color=color_struct,
        edgecolor="black",
        linewidth=0.3
    )
    bottom = [b + h for b, h in zip(bottom, struct_pct)]

    # Percent labels inside each stacked part
    for i in range(len(processed_segments)):
        m = miss_pct[i]
        s = struct_pct[i]
        if m > 0:
            # Centered in the bottom segment
            plt.text(
                i, m / 2.0, f"{m:.1f}%",
                ha="center", va="center",
                fontsize="xx-small",
                color="white"
            )
        if s > 0:
            # Centered in the structural segment
            plt.text(
                i, m + (s / 2.0), f"{s:.1f}%",
                ha="center", va="center",
                fontsize="xx-small",
                color="black"
            )

    # Annotate stacked fraction (above bar) and overall % (above that)
    for i, seg in enumerate(processed_segments):
        filtered_sum = miss_per_seg.get(seg, 0) + struct_per_seg.get(seg, 0)
        total_defs = total_per_seg.get(seg, 0)
        top = bottom[i]
        if top > 0 and total_defs > 0:
            overall_pct = (filtered_sum / total_defs) * 100.0

            # Stacked fraction
            plt.text(
                i,
                top + 1.0,
                f"{filtered_sum}\n—\n{total_defs}",
                ha="center",
                va="bottom",
                fontsize="xx-small"
            )

            # Overall % above that
            plt.text(
                i,
                top + 8.0,
                f"{overall_pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize="xx-small",
                fontweight="bold"
            )

    # Calculate overall mean percentage
    mean_pct = sum(
        ((miss_per_seg.get(s, 0) + struct_per_seg.get(s, 0)) / total_per_seg.get(s, 1)) * 100.0
        for s in processed_segments
    ) / len(processed_segments)

    plt.axhline(
        y=mean_pct,
        color="#CA4B2E",  # Light red from standard colour palette
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        zorder=0
    )
    plt.text(
        x=- 0.3,
        y=mean_pct + 0.5,
        s=f"{mean_pct:.1f}%",
        color="#CA4B2E",  # Light red from standard colour palette
        fontsize="xx-small",
        fontweight="bold",
        va="bottom",
        ha="right",
        alpha=0.8
    )

    # Layout
    max_ratio = max(bottom) if bottom else 0
    plt.ylim(0, max_ratio * 1.2)

    plt.xlabel("ECLASS Segments")
    plt.ylabel("Non-semantic ECLASS Definition Ratio")
    plt.title("Non-semantic ECLASS Definitions per Segment")
    plt.xticks(rotation=45)

    plt.legend(
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False
    )

    plt.grid(axis="y", linestyle="--", color="black", alpha=0.2)

    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Initialising non-semantic definition plotter ...")
    segments = list(range(13, 52)) + [90]
    input_path = "../../data/original-classes/eclass-{segment}.csv"
    output_path = "../../visualisation/nonsemantic_definitions_ratio.png"

    # Run for each segment
    plot_nonsemantic_ratios(input_path, segments, exceptions, output_path, logger)
