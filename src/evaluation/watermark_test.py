import os
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from src.embedding.filter import (filter_definitions_missing, filter_definitions_missing_suffix, filter_definitions_structural)

HIDDEN_CHARS_MAP = {
    "\u00A0": "NO-BREAK SPACE",
    "\u200B": "ZERO WIDTH SPACE",
    "\u200C": "ZERO WIDTH NON-JOINER",
    "\u200D": "ZERO WIDTH JOINER",
    "\u202F": "NARROW NO-BREAK SPACE",
    "\u2060": "WORD JOINER",
    "\uFEFF": "ZERO WIDTH NO-BREAK SPACE",
}

MISSING_SET = frozenset(filter_definitions_missing)
MISSING_SUFFIXES = tuple(filter_definitions_missing_suffix)
STRUCTURAL_SET = frozenset(filter_definitions_structural)


CHAR_SLUGS = {
    char: f"U{ord(char):04X}" for char in HIDDEN_CHARS_MAP.keys()
}

MODEL_NAMES = [
    'mboth/distil-eng-quora-sentence',
    'BAAI/bge-large-en-v1.5'
]


def load_and_sample_definitions(
        csv_path: str,
        n_samples: int,
        definition_col_index: int,
        missing_set: frozenset,
        structural_set: frozenset,
        missing_suffixes: tuple[str]
) -> pd.DataFrame:
    """
    Loads a CSV, identifies the definition column by index,
    filters out rows based on missing_set, structural_set, OR missing_suffixes,
    and returns a random sample of n rows.
    """
    # require: Check if the file exists before trying to read
    filepath = Path(csv_path)
    assert filepath.exists(), f"Input file not found at: {csv_path}"

    # require: Must sample at least one definition
    assert n_samples > 0, "n_samples must be greater than 0"

    # require: missing_suffixes must be a tuple for .str.endswith()
    assert isinstance(missing_suffixes, tuple), "missing_suffixes must be a tuple"

    df = pd.read_csv(filepath)

    # require: Check if the specified column index is valid
    assert definition_col_index < len(df.columns), \
        f"Column index {definition_col_index} is out of bounds."

    # Get the column name from its index (e.g., index 2 is the 3rd column)
    definition_col_name = df.columns[definition_col_index]

    # --- NEW: Combined Filtering ---
    initial_count = len(df)

    # Get the series of definitions, ensuring it's string type and NaNs are empty strings
    definitions_series = df[definition_col_name].astype(str).fillna('')

    # Create a mask for each "bad" condition
    mask_in_missing_set = definitions_series.isin(missing_set)
    mask_in_structural_set = definitions_series.isin(structural_set)
    mask_is_suffix = definitions_series.str.endswith(missing_suffixes)

    # Combine them with OR (|): throw out if *any* condition is true
    combined_bad_mask = (
            mask_in_missing_set | mask_in_structural_set | mask_is_suffix
    )

    # Keep only the rows that are NOT (~) in the bad mask
    df_filtered = df[~combined_bad_mask]

    filtered_count = len(df_filtered)
    print(f"    Loaded {initial_count} rows. Filtered out {initial_count - filtered_count} unwanted definitions.")
    # ---

    # require: Check if we have enough definitions to sample from *after* filtering
    assert filtered_count >= n_samples, \
        f"Not enough rows in CSV after filtering ({filtered_count}) to sample {n_samples}"

    # Take a random sample from the filtered DataFrame
    sampled_df = df_filtered.sample(n=n_samples, random_state=42)  # Using a fixed state

    # Rename the target column to a standard 'definition' for easier processing
    sampled_df = sampled_df.rename(
        columns={definition_col_name: 'definition'}
    )

    # ensure: The output DataFrame should have the correct number of rows
    assert len(sampled_df) == n_samples, "Sampled DataFrame size is incorrect"

    return sampled_df


def add_perturbations(
        df: pd.DataFrame,
        hidden_chars: dict
) -> pd.DataFrame:
    """
    Adds 7 new 'perturbed_...' columns to the DataFrame.

    For each row, it picks one random space. It then creates 7
    perturbed versions, each replacing that *same* space with one
    of the 7 hidden characters.
    """
    # require: The input DataFrame must have a 'definition' column
    assert 'definition' in df.columns, "DataFrame missing 'definition' column"

    # Create empty lists to hold the new column data
    new_cols_data = {
        f"perturbed_{CHAR_SLUGS[char]}": [] for char in hidden_chars.keys()
    }
    replaced_indices = []

    # Iterate over each definition in the DataFrame
    for definition in df['definition']:
        # Find all indices of standard spaces
        space_indices = [i for i, char in enumerate(definition) if char == ' ']

        if not space_indices:
            # --- Edge Case: No spaces ---
            # Can't perturb, so just add the original text to all columns
            replaced_indices.append(-1)  # Use -1 to signify no replacement
            for char_code in hidden_chars.keys():
                slug = CHAR_SLUGS[char_code]
                col_name = f"perturbed_{slug}"
                new_cols_data[col_name].append(definition)
        else:
            # --- Standard Case: Found spaces ---
            # Pick one random, consistent space index
            index_to_replace = random.choice(space_indices)
            replaced_indices.append(index_to_replace)

            # Split the string at that index
            prefix = definition[:index_to_replace]
            suffix = definition[index_to_replace + 1:]

            # Create all 7 perturbations
            for char_code in hidden_chars.keys():
                slug = CHAR_SLUGS[char_code]
                col_name = f"perturbed_{slug}"

                # Build the new perturbed string
                perturbed_text = prefix + char_code + suffix
                new_cols_data[col_name].append(perturbed_text)

                # invariant: The new text must not contain the original space
                # at that index
                assert perturbed_text[index_to_replace] == char_code

    # Add the new columns to the DataFrame
    df['replaced_space_index'] = replaced_indices
    for col_name, data in new_cols_data.items():
        df[col_name] = data

    # ensure: The new columns must have been created
    for char_code in hidden_chars.keys():
        col_name = f"perturbed_{CHAR_SLUGS[char_code]}"
        assert col_name in df.columns, f"Failed to create column: {col_name}"

    return df


def get_embeddings(
        texts: list[str],
        model_name: str,
        device: str
) -> np.ndarray:
    """
    Generates sentence embeddings for a list of texts using a specified model.
    """
    # require: We must have texts to embed
    assert texts is not None and len(texts) > 0, "Input text list is empty"

    print(f"    Loading model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    print(f"    Generating {len(texts)} embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # ensure: The output shape must match the input count
    assert embeddings.shape[0] == len(texts), "Embedding output shape mismatch"

    return embeddings


def compare_embeddings(
        embeddings_orig: np.ndarray,
        embeddings_pert: np.ndarray
) -> np.ndarray:
    """
    Calculates the element-wise (diagonal) cosine similarity
    between two embedding arrays.
    """
    # require: Both arrays must be 2D and have the same shape
    assert embeddings_orig.ndim == 2, "Original embeddings not 2D"
    assert embeddings_pert.ndim == 2, "Perturbed embeddings not 2D"
    assert embeddings_orig.shape[0] == embeddings_pert.shape[0], \
        "Embedding array row counts do not match"
    # Note: We no longer require identical shape, just identical row count.

    # Calculate pairwise similarity (e.g., orig_1 vs pert_1, orig_1 vs pert_2)
    similarity_matrix = cosine_similarity(embeddings_orig, embeddings_pert)

    # We only care about the diagonal: (orig_1 vs pert_1), (orig_2 vs pert_2)
    scores = similarity_matrix.diagonal()

    # ensure: The number of scores must match the number of embeddings
    assert len(scores) == embeddings_orig.shape[0], "Score count mismatch"

    return scores


def save_similarity_heatmap(
        scores_df: pd.DataFrame,
        model_name: str,
        output_path: str
):
    """
    Generates and saves a heatmap of the similarity scores.

    Args:
        scores_df: A DataFrame where rows are sentences and
                   columns are perturbation types.
        model_name: The name of the model (for the title).
        output_path: The full path to save the PNG file.
    """
    # require: The DataFrame must not be empty
    assert not scores_df.empty, "scores_df is empty, cannot plot heatmap."

    # Set the y-axis labels to be "Sentence 0", "Sentence 1", etc.
    sentence_labels = [f"Sentence {i}" for i in scores_df.index]

    plt.figure(figsize=(12, 8))  # Adjust size as needed

    # We use a "reverse" colormap (like 'rocket_r') so that
    # lower scores (worse) are darker and higher scores (better) are lighter.
    # vmax=1.0 anchors the colorbar, ensuring 1.0 is the lightest color.
    heatmap = sns.heatmap(
        scores_df,
        annot=True,  # Show the scores in each cell
        fmt=".6f",  # Format scores to 6 decimal places
        cmap="rocket_r",  # Use a reverse colormap (dark=low, light=high)
        vmax=1.0,  # Set 1.0 as the "best" (lightest) color
        yticklabels=sentence_labels,
    )

    heatmap.set_title(
        f"Similarity Scores: Original vs. Perturbed\nModel: {model_name}",
        fontdict={'fontsize': 16, 'fontweight': 'bold'}
    )
    heatmap.set_xlabel("Perturbation Character", fontsize=12)
    heatmap.set_ylabel("Sentence", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
    plt.tight_layout()  # Fit the plot neatly into the figure

    # ensure: The output path must have a directory that exists
    save_dir = Path(output_path).parent
    assert save_dir.exists(), f"Output directory does not exist: {save_dir}"

    plt.savefig(output_path)
    print(f"    Heatmap saved to: {output_path}")
    plt.close()  # Close the figure to free up memory


# --- Main Execution (MODIFIED) ---

def main():
    """
    Orchestrates the full experiment:
    1. Load and sample data
    2. Add 7 perturbations for each definition
    3. Save the new wide-format dataset
    4. For each model:
        a. Generate embeddings for original texts
        b. For each of the 7 perturbation types:
           i. Generate embeddings for perturbed texts
           ii. Save embeddings
           iii. Store similarity scores
        c. Report stats grouped by Perturbation
        d. Report stats grouped by Sentence
    """
    # --- 1. Configuration ---
    N_TO_SAMPLE = 10
    DEFINITION_COLUMN_INDEX = 2  # The 3rd column

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Navigate from src/evaluation/ to the project root (two levels up)
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    # Now build paths relative to project root
    INPUT_DIR = PROJECT_ROOT / "data" / "extracted"
    OUTPUT_DIR = PROJECT_ROOT / "experiment_data"

    # !! IMPORTANT: Change this variable to the name of your file !!
    INPUT_CSV_NAME = "eclass-all.csv"  # Based on your error message

    INPUT_CSV_PATH = INPUT_DIR / INPUT_CSV_NAME
    OUTPUT_CSV_PATH = OUTPUT_DIR / "sampled_definitions.csv"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- File path validation ---
    if not INPUT_CSV_PATH.exists():
        print(f"\n--- Error ---")
        print(f"Input file not found at the expected path:")
        print(f"  {INPUT_CSV_PATH}")

        # Check for other CSVs in the directory
        if not INPUT_DIR.exists():
            print(f"\nThe directory {INPUT_DIR} doesn't seem to exist.")
            print("Please check your path.")
            print("\nProject structure detected:")
            print(f"  Script location: {SCRIPT_DIR}")
            print(f"  Project root:    {PROJECT_ROOT}")
            print(f"  Expected input:  {INPUT_DIR}")
            print("Experiment aborted.")
            return  # Stop execution

        other_csvs = list(INPUT_DIR.glob("*.csv"))
        if other_csvs:
            print("\nHowever, I found these CSV files in that directory:")
            for f in other_csvs:
                print(f"  - {f.name}")
            print(f"\nTo fix this, please update the 'INPUT_CSV_NAME' variable")
            print("inside this script to match the file you want to use.")
        else:
            print(f"\nNo CSV files were found in {INPUT_DIR} at all.")

        print("Experiment aborted.")
        return  # Stop execution

    # --- 2. Setup Device ---
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():  # For your Mac
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # --- 3. Load, Sample, and Perturb Data ---
    print(f"Loading and sampling {N_TO_SAMPLE} definitions...")
    try:
        df_sampled = load_and_sample_definitions(
            str(INPUT_CSV_PATH), N_TO_SAMPLE, DEFINITION_COLUMN_INDEX, MISSING_SET, STRUCTURAL_SET, MISSING_SUFFIXES
        )

        print("Adding all 7 perturbations...")
        df_final = add_perturbations(df_sampled, HIDDEN_CHARS_MAP)

        # --- 4. Save New CSV ---
        df_final.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved {N_TO_SAMPLE} sampled definitions (with 7 perturbations each) to: {OUTPUT_CSV_PATH}")

        # Prepare the *original* definitions list (this is done once)
        definitions_original = df_final['definition'].tolist()

        # --- 5. Generate, Save, and Compare Embeddings ---
        for model_name in MODEL_NAMES:
            print("\n" + "=" * 70)
            print(f"Processing Model: {model_name}")
            print("=" * 70)

            # --- Get Original Embeddings (Once per model) ---
            print("  Generating embeddings for ORIGINAL texts...")
            model_slug = model_name.split('/')[-1]  # e.g. "bge-large-en-v1.5"
            embed_orig = get_embeddings(
                definitions_original, model_name, device
            )

            # Save original embeddings
            orig_path = OUTPUT_DIR / f"{model_slug}_orig.npy"
            np.save(orig_path, embed_orig)
            print(f"  Saved original embeddings to: {orig_path}")

            # --- Loop over all 7 perturbation types ---

            # This dict will store scores like: {'U00A0': [0.99, 0.98,...], ...}
            model_perturbation_scores = {}

            print("\n  --- 1. Results (Grouped by Perturbation) ---")

            for char_code, char_name in HIDDEN_CHARS_MAP.items():
                slug = CHAR_SLUGS[char_code]
                col_name = f"perturbed_{slug}"

                print(f"\n  Perturbation: {char_name} ({slug})")

                # Get the list of texts for this specific perturbation
                definitions_perturbed = df_final[col_name].tolist()

                # Generate embeddings
                embed_pert = get_embeddings(
                    definitions_perturbed, model_name, device
                )

                # Save embeddings
                pert_path = OUTPUT_DIR / f"{model_slug}_{col_name}.npy"
                np.save(pert_path, embed_pert)

                # Compare and get the 10 scores for this perturbation
                similarity_scores = compare_embeddings(embed_orig, embed_pert)

                # Store scores for later
                model_perturbation_scores[slug] = similarity_scores

                # Report stats for *this perturbation* across all 10 sentences
                print(f"    Avg similarity (across {N_TO_SAMPLE} sentences):")
                print(f"      Mean:   {np.mean(similarity_scores):.8f}")
                print(f"      Median: {np.median(similarity_scores):.8f}")
                print(f"      Min:    {np.min(similarity_scores):.8f}")
                print(f"      Max:    {np.max(similarity_scores):.8f}")
                print(f"      StdDev: {np.std(similarity_scores):.8f}")

            # --- 6. Report Averages by Sentence ---

            print("\n  --- 2. Results (Grouped by Sentence) ---")

            # Convert the scores dict to a DataFrame
            # Rows = Sentences (0-9), Columns = Perturbations (U00A0, U200B, ...)
            scores_df = pd.DataFrame(model_perturbation_scores)

            heatmap_path = OUTPUT_DIR / f"{model_slug}_similarity_heatmap.png"
            save_similarity_heatmap(scores_df, model_name, str(heatmap_path))

            # Calculate the mean for each sentence (row) across all 7 perturbations
            avg_by_sentence = scores_df.mean(axis=1)  # axis=1 means across columns

            print(f"    Avg similarity (across all 7 perturbations):")
            print(f"      Overall Mean:   {avg_by_sentence.mean():.8f}")
            print(f"      Median:         {avg_by_sentence.median():.8f}")
            print(f"      Min (worst sentence): {avg_by_sentence.min():.8f}")
            print(f"      Max (best sentence):  {avg_by_sentence.max():.8f}")
            print(f"      StdDev:         {avg_by_sentence.std():.8f}")

            print("\n    Average similarity for each sentence:")
            for i, score in enumerate(avg_by_sentence):
                print(f"      Sentence {i:02d}: {score:.8f}")

    except AssertionError as e:
        print(f"\nError: {e}")
        print("Experiment aborted.")
    except FileNotFoundError:
        print(f"\nError: Could not find input file.")
        print(f"Please check that INPUT_CSV_PATH is correct.")


if __name__ == "__main__":
    main()