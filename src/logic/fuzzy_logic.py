"""Module to construct and check transitivity characteristics for fuzzy graphs."""

import numpy as np
import skfuzzy as fuzz
from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from src.utils.logger import LoggerFactory


def k_maxprod_composition(r: np.ndarray, k: int = 2) -> np.ndarray:
    """Compute the k-hop max–product composition of a fuzzy graph."""

    result = np.array(r, dtype=float)
    for _ in range(1, k):
        result = fuzz.maxprod_composition(result, r)
    return result


def is_maxprod_transitive(r: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if a fuzzy graph is max–product-transitive."""

    r = np.array(r, dtype=float)
    rr = fuzz.maxprod_composition(r, r)
    return np.all(r + tol >= rr)  # Check r >= r x r


def maxprod_transitive_closure(
        r: np.ndarray,
        tol: float = 1e-12,
) -> np.ndarray:
    """Return the smallest max–product-transitive superset of a fuzzy graph."""

    r = np.array(r, dtype=float)
    max_iter = r.shape[0] - 1  # Worst case: longest path in n nodes has n-1 edges

    for _ in range(max_iter):
        rr = fuzz.maxprod_composition(r, r)
        r_new = np.maximum(r, rr)  # S-norm update, r = r ∨ (r x r)
        if np.allclose(r_new, r, atol=tol):
            break  # Converged
        r = r_new
    return r


def transitivity_violations(
        r: np.ndarray,
        k: int = 2,
        tol: float = 1e-12,
) -> dict[tuple[int, int], tuple[float, float]]:  # {(i,j): (direct score, path score)}
    """Return a list of transitivity violations with a maximum path length k for a given fuzzy graph."""

    r = np.array(r, dtype=float)
    rr = k_maxprod_composition(r, k)  # r^(k)

    # Find violations
    mask = (r + tol) < rr
    n = r.shape[0]
    i, j = np.indices((n, n))
    mask &= i < j  # Mask because symmetric matrix

    # Prepare output
    idx_i, idx_j = np.where(mask)
    out = {}
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        key = (i, j)
        direct_score = float(r[i, j])
        via_k_score = float(rr[i, j])
        out[key] = (direct_score, via_k_score)

    # Log violation percentage of all matches
    total_violations = len(out)
    total_count = (r.shape[0] * (r.shape[0] - 1)) / 2
    total_percentage = (total_violations / total_count) * 100 if total_count else 0.0
    logger = LoggerFactory.get_logger(__name__)
    logger.info(
        "Violations: %d / %d ≈ %s%%",
        total_violations, total_count, total_percentage
    )

    return out


def k_exact_simple_paths(
        r: np.ndarray,
        k: int,
        start_nodes: set[int],
        end_nodes: set[int]
) -> dict[tuple[int, int], tuple[float, list[int]]]:  # {(i,j): (path score, path)}
    """
    Return a list of simple paths with the exact length k and the best possible k-hop matching score from predefined
    starting nodes in a fuzzy graph.
    """

    # Setup
    r = np.array(r, dtype=float)
    n = r.shape[0]
    best_simple_paths = {}

    # Depth-first search
    def dfs(current_node: int, depth: int, score: float, start_node: int) -> None:
        if depth == k:  # Found simple path of length k
            if current_node in end_nodes:
                key = (int(start_node), int(current_node))
                candidate = (float(score), path.copy())
                if key not in best_simple_paths or score > best_simple_paths[key][0]:  # Found path with higher score
                    best_simple_paths[key] = candidate
            return
        if score == 0.0:
            return
        row = r[current_node]  # Neighbours
        for v in np.flatnonzero(row > 0.0):  # Run into all neighbours
            if seen[v]:
                continue
            weight = row[v]
            seen[v] = True
            path.append(int(v))
            dfs(v, depth + 1, score * weight, start_node)
            path.pop()
            seen[v] = False

    # Start depth-first search for every starting node
    for s in tqdm(iterable=start_nodes, total=len(start_nodes), desc="Starts"):
        seen = [False] * n
        seen[s] = True
        path = [s]
        dfs(s, 0, 1.0, s)
    return best_simple_paths


def best_simple_path(
        r: np.ndarray,
        start_node: int,
        end_node: int,
        tol: float = 1e-8
) -> (float, list[int]):  # (path score, path)
    """Return a simple path with the best possible matching score in a fuzzy graph."""

    r = np.array(r, dtype=float)

    # Shift fuzzy graph weights into logarithmic space for Dijkstra
    r = np.clip(r, tol, 1 - tol, r)  # Avoid  logarithmic singularity at zero and deleting paths with weight 1
    r_log = -np.log(r)

    # Run Dijkstra
    distances, predecessors = dijkstra(csgraph=r_log, directed=False, indices=start_node, return_predecessors=True)

    # Reconstruct path
    path = [end_node]
    while path[-1] != start_node:
        path.append(int(predecessors[path[-1]]))
    path.reverse()
    score = float(np.exp(-distances[end_node]))

    return score, path
