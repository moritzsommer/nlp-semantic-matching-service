"""
This module can calculate the matching scores using cosine similarity
and stores results in an SQLite database.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np

def compute_similarities(embeddings_dir: str, output_db: str) -> None:
    file_paths: List[Path] = list(Path(embeddings_dir).glob("*.json"))

    entries: List[Dict[str, object]] = []
    for path in tqdm(file_paths, desc="Loading entries into memory"):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
            for entry in data:
                entries.append({
                    "id": entry["id"],
                    "vector_norm": entry["vector_norm"],
                    "embedding": entry["embedding"]
                })

    conn = sqlite3.connect(output_db)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS similarities (
            id1 TEXT,
            id2 TEXT,
            cosine_similarity REAL
        )
    """)
    conn.commit()

    for i in tqdm(range(len(entries)), desc="     Computing similarities"):
        id1 = entries[i]["id"]
        vec1 = entries[i]["embedding"]
        norm1 = entries[i]["vector_norm"]

        for j in range(i + 1, len(entries)):
            id2 = entries[j]["id"]
            vec2 = entries[j]["embedding"]
            norm2 = entries[j]["vector_norm"]

            sim: float = float(np.dot(vec1, vec2) / (norm1 * norm2))
            cur.execute("INSERT INTO similarities (id1, id2, cosine_similarity) VALUES (?, ?, ?)",
                        (id1, id2, sim))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    compute_similarities(
        "../../data/embeddings/qwen3_eclass_14_and_15",
        "../../data/scores/scores_qwen3_eclass_14_and_15.sqlite"
    )
