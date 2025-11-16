"""Module to embed ECLASS definitions using the gemini-embedding-001 transformer model."""

import logging

import numpy as np
import pandas as pd
from google import genai
from google.genai.types import EmbedContentConfig

from src.utils.json_io import save_json
from src.utils.logger import LoggerFactory


def embed_eclass(
        input_path: str,
        output_path: str,
        logger: logging.Logger
) -> None:
    """Loads ECLASS data, computes embeddings with Gemini and saves the results as JSON."""

    logger.info(f"Starting embedder ...")

    # Load ECLASS classes from CSV file
    try:
        database = pd.read_csv(input_path, sep=",")
        logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
    except Exception as e:
        logger.error(f"Failed to read file: {input_path}, Error: {e}")
        return

    # Ensure required columns exist
    required_columns = ["definition", "preferred-name", "id"]
    for col in required_columns:
        if col not in database.columns:
            logger.error(f"Missing required column: {col}")
            return

    # Compute embeddings, no batch mode
    definitions = database["definition"].tolist()
    embeddings = []
    logger.info("Computing embeddings ...")
    for definition in definitions:
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=definition,
                config=EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=3072
                )
            )
            embeddings.append(result.embeddings[0].values)
        except Exception as e:
            logger.error(f"Error while embedding: {e}")
            continue
        if len(embeddings) % 10 == 0:
            logger.info(f"Processed {len(embeddings)} entries.")

    # Save results
    embedded_entries = [
        {
            "id": row["id"],
            "preferred-name": row["preferred-name"],
            "definition": row["definition"],
            "vector-norm": float(np.linalg.norm(embedding)),
            "embedding": embedding,
        }
        for (_, row), embedding in zip(database.iterrows(), embeddings)
    ]
    logger.info(f"Embedding computation finished for {input_path}. Total processed: {len(embedded_entries)}")
    save_json(embedded_entries, output_path)
    logger.info(f"Saved embeddings to: {output_path}")


if __name__ == "__main__":
    # Settings
    run_per_segment = True  # Compute embeddings per segment or in one run
    exceptions = []  # Exclude specific segments if run per segment
    project = "PLACEHOLDER"  # Project name from Google Cloud

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Initialising embedder ...")

    # Connect Gemini client via Vertex Ai
    client = genai.Client(vertexai=True, project=project, location="global")
    logger.info(f"Client loaded.")

    if run_per_segment:
        # Run for each segment
        segments = [46]
        for segment in segments:
            if segment in exceptions:
                logger.warning(f"Skipping segment {segment}.")
                continue
            input_path = f"../../data/extracted-classes/3-normalised-classes/eclass-{segment}.csv"
            output_path = f"../../data/embedded-classes/eclass-{segment}-embeddings-gemini.json"
            embed_eclass(input_path, output_path, logger)
    else:
        # Run for combined segments
        input_path = f"../../data/extracted-classes/3-normalised-classes/eclass-0.csv"
        output_path = f"../../data/embedded-classes/eclass-0-embeddings-gemini.json"
        embed_eclass(input_path, output_path, logger)
