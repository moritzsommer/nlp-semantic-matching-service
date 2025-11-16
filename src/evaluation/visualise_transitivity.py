"""Module to plot transitivity characteristics for fuzzy graphs."""

import numpy as np
from matplotlib import pyplot as plt

from src.embedding.scores import load_matrix
from src.service.fuzzy_logic import k_exact_simple_paths, k_maxprod_composition


def plot_direct_vs_k_maxprod_composition(
        r: np.ndarray,
        k: int,
        output_path: str,
        model: str,
        segment: str,
        tol: float = 1e-12
) -> None:
    """Plot all direct vs k-hop max–product (paths with maximum length k) matching scores for a given fuzzy graph."""

    r = np.array(r, dtype=float)
    rr = k_maxprod_composition(r, k)  # r^(k)

    # Mask because symmetric matrix
    n = r.shape[0]
    i, j = np.indices((n, n))
    mask = (i < j)
    x = r[mask]
    y = rr[mask]

    # Find violations
    viol = (x + tol) < y
    delta = y - x
    d_viol = delta[viol]
    b_max = float(d_viol.max()) if d_viol.size else 0.0
    b_min = float(delta.min()) if delta.size else 0.0

    # Plotting
    plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.gca()

    colour_violations = "#EFC47A"  # Light golden shade from the standard colour palette
    colour_non_violations = "#7D91A6"  # Light blue from standard colour palette

    # Points
    ax.scatter(x[viol], y[viol], s=8, alpha=0.25, label="Violations", color=colour_violations)
    ax.scatter(x[~viol], y[~viol], s=8, alpha=0.25, label="Non-violations", color=colour_non_violations)

    # Reference lines
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.6, label="y = x", color="black", alpha=1.0)
    if b_max > 0:
        x_end = max(0.0, 1.0 - b_max)
        ax.plot([0, x_end], [b_max, b_max + x_end], linestyle="-.", linewidth=1.0,
                label=f"y = x + {b_max:.4g}", color=colour_violations, alpha=1)
    if b_min < 0:
        x_end = min(1.0, 1.0 - b_min)
        ax.plot([0, x_end], [b_min, b_min + x_end], linestyle="-.", linewidth=1.0,
                label=f"y = x - {abs(b_min):.4g}", color=colour_non_violations, alpha=1)

    # Title and labels
    ax.set_title(
        f"ECLASS Semantic Transitivity - Embedding Model {model.capitalize()} - Segment {segment}",
        pad=15
    )
    ax.set_xlabel("Direct Matching Score")
    ax.set_ylabel(f"Best Maximum {k}-hop Matching Score")

    # Limits, axes and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="y", linestyle="--", color="black", alpha=0.1)
    ax.grid(axis="x", linestyle="--", color="black", alpha=0.1)

    # Legend style
    legend = ax.legend(
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False
    )
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(1.0)

    # Save
    plt.savefig(output_path, bbox_inches="tight")


def plot_direct_vs_k_exact_simple_paths(
        r: np.ndarray,
        k: int,
        output_path: str,
        model: str,
        segment: str,
        tol: float = 1e-12,
) -> None:
    """Plot all direct vs k-hop simple path (paths with exact length k) matching scores for a given fuzzy graph."""

    r = np.array(r, dtype=float)
    n = r.shape[0]

    # Compute paths with exact length k and save in fuzzy graph
    rr = np.zeros_like(r, dtype=float)
    for (start, end), (score, path) in k_exact_simple_paths(r, k, set(range(n)), set(range(n))).items():
        rr[start, end] = float(score)

    # Mask because symmetric matrix
    i, j = np.indices((n, n))
    mask = (i < j)
    x = r[mask]
    y = rr[mask]

    # Find violations
    viol = (x + tol) < y
    delta = y - x
    d_viol = delta[viol]
    b_max = float(d_viol.max()) if d_viol.size else 0.0
    b_min = float(delta.min()) if delta.size else 0.0

    # Plotting
    plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.gca()

    colour_violations = "#EFC47A"  # Light golden shade from the standard colour palette
    colour_non_violations = "#7D91A6"  # Light blue from standard colour palette

    # Points
    ax.scatter(x[viol], y[viol], s=8, alpha=0.25, label="Violations", color=colour_violations)
    ax.scatter(x[~viol], y[~viol], s=8, alpha=0.25, label="Non-violations", color=colour_non_violations)

    # Reference lines
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.6, label="y = x", color="black", alpha=1.0)
    if b_max > 0:
        x_end = max(0.0, 1.0 - b_max)
        ax.plot([0, x_end], [b_max, b_max + x_end], linestyle="-.", linewidth=1.0,
                label=f"y = x + {b_max:.4g}", color=colour_violations, alpha=1)
    if b_min < 0:
        x_end = min(1.0, 1.0 - b_min)
        ax.plot([0, x_end], [b_min, b_min + x_end], linestyle="-.", linewidth=1.0,
                label=f"y = x - {abs(b_min):.4g}", color=colour_non_violations, alpha=1)

    # Title and labels
    ax.set_title(
        f"ECLASS Semantic Transitivity - Embedding Model {model.capitalize()} - Segment {segment}",
        pad=15
    )
    ax.set_xlabel("Direct Matching Score")
    ax.set_ylabel(f"Best Exact {k}-hop Matching Score")

    # Limits, axes and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="y", linestyle="--", color="black", alpha=0.1)
    ax.grid(axis="x", linestyle="--", color="black", alpha=0.1)

    # Legend style
    legend = ax.legend(
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False
    )
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(1.0)

    # Save
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    # Settings
    max_hops = 2  # Maximum path length
    segment = 46  # Segment in the range [13, 52] and 90
    transformer = "qwen3"  # Cluster embeddings from this transformer (qwen3, bge, gemini)

    # Run
    m, row_idx, col_idx = load_matrix(
        f"../../data/scores/eclass-scores-{segment}-{transformer}.sqlite",
        table="similarities",
        dtype=np.float64,
        chunk_size=200_000
    )

    plot_direct_vs_k_maxprod_composition(
        r=m,
        k=max_hops,
        output_path=f"../../visualisation/eclass-{segment}-direct-vs-max-{max_hops}-hop-{transformer}",
        model=transformer,
        segment=str(segment)
    )

    plot_direct_vs_k_exact_simple_paths(
        r=m,
        k=max_hops,
        output_path=f"../../visualisation/eclass-{segment}-direct-vs-exact-{max_hops}-hop-{transformer}",
        model=transformer,
        segment=str(segment)
    )
