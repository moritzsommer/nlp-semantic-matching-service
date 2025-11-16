"""Module to embed ECLASS definitions using the BAAI/bge-large-en-v1.5 transformer model."""

from embedder import EclassEmbedder

if __name__ == "__main__":
    # Settings
    run_per_segment = False  # Compute embeddings per segment or in one run
    exceptions = []  # Exclude specific segments if run per segment

    # Setup
    embedder = EclassEmbedder(model_name="BAAI/bge-large-en-v1.5")

    if run_per_segment:
        # Run for each segment
        segments = list(range(13, 52)) + [90]
        for segment in segments:
            if segment in exceptions:
                embedder.logger.warning(f"Skipping segment {segment}.")
                continue
            input_path = f"../../data/extracted-classes/3-normalised-classes/eclass-{segment}.csv"
            output_path = f"../../data/embedded-classes/eclass-{segment}-embeddings-bge.json"
            embedder.embed_eclass(input_path, output_path)
    else:
        # Run for combined segments
        input_path = f"../../data/extracted-classes/3-normalised-classes/eclass-0.csv"
        output_path = f"../../data/embedded-classes/eclass-0-embeddings-bge.json"
        embedder.embed_eclass(input_path, output_path)
