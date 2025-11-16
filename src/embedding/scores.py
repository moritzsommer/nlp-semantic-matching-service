"""This module can calculate the matching scores using cosine similarity and stores results in an SQLite database."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def compute_similarities(embeddings_dir: str, embeddings_model: str, embeddings_segment: int, output_dir: str) -> None:
    """Computes cosine similarities between all embedding pairs and stores them in an SQLite database."""

    file_ending = f"*{embeddings_segment}-embeddings-{embeddings_model}.json"
    file_paths: List[Path] = list(Path(embeddings_dir).glob(file_ending))

    entries: List[Dict[str, object]] = []
    for path in tqdm(file_paths, desc="Loading entries into memory"):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
            for entry in data:
                entries.append({
                    "id": entry["id"],
                    "vector-norm": entry["vector-norm"],
                    "embedding": entry["embedding"]
                })

    output_path = f"{output_dir}eclass-scores-{embeddings_segment}-{embeddings_model}.sqlite"
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS similarities")
    cur.execute("""
                CREATE TABLE similarities
                (
                    id1               TEXT,
                    id2               TEXT,
                    cosine_similarity REAL
                )
                """)
    conn.commit()

    for i in tqdm(range(len(entries)), desc="Computing similarities"):
        id1 = entries[i]["id"]
        vec1 = entries[i]["embedding"]
        norm1 = entries[i]["vector-norm"]

        for j in range(i, len(entries)):
            id2 = entries[j]["id"]
            vec2 = entries[j]["embedding"]
            norm2 = entries[j]["vector-norm"]

            sim: float = float(np.dot(vec1, vec2) / (norm1 * norm2))
            cur.execute("INSERT INTO similarities (id1, id2, cosine_similarity) VALUES (?, ?, ?)",
                        (id1, id2, sim))

    conn.commit()
    conn.close()


def load_matrix(
        input_path: str,
        table: str = "similarities",
        dtype=np.float32,
        chunk_size: int = 200_000
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Load a similarity matrix from a SQLite database, preserving row/column order by insertion."""

    conn = sqlite3.connect(input_path)
    cur = conn.cursor()

    # Pass 1: Discover row/column ids in the order they first appear
    row_idx: Dict[str, int] = {}
    col_idx: Dict[str, int] = {}
    rows: List[str] = []
    cols: List[str] = []

    cur.execute(f"SELECT id1, id2 FROM {table} ORDER BY rowid")
    while True:
        chunk = cur.fetchmany(chunk_size)
        if not chunk:
            break
        for a, b in chunk:
            if a not in row_idx:
                row_idx[a] = len(rows)
                rows.append(a)
            if b not in col_idx:
                col_idx[b] = len(cols)
                cols.append(b)

    m = np.full((len(rows), len(cols)), 0.0, dtype=np.float32)

    # Pass 2: Fill values
    cur.execute(f"SELECT id1, id2, cosine_similarity FROM {table} ORDER BY rowid")
    while True:
        chunk = cur.fetchmany(chunk_size)
        if not chunk:
            break
        for a, b, sim in chunk:
            i = row_idx[a]  # Always present from pass 1
            j = col_idx[b]
            m[i, j] = np.float32(sim) if dtype == np.float32 else float(sim)

    conn.close()

    # Adapt similarity scores for our scenario
    m = m + m.T - np.diag(np.diag(m))  # Undirected Graphs
    np.fill_diagonal(m, 1.0)  # Reflexivity
    np.clip(m, 0.0, 1.0, m)  # Counteract rounding
    return m, row_idx, col_idx


if __name__ == '__main__':
    compute_similarities(
        "../../data/embedded-classes/",
        "gemini",  # qwen3, bge, gemini
        46,
        "../../data/scores/"
    )
