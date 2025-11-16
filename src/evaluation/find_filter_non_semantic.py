"""Module to identify potentially non-semantic ECLASS definitions based on regex patterns."""

import re

import pandas as pd

from src.utils.logger import LoggerFactory

if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments

    # Setup
    logger = LoggerFactory.get_logger(__name__)

    # Patterns that could occur in non-semantic ECLASS definitions
    no_def_pattern = re.compile(r"(definition|tbd|determined|defined)", re.IGNORECASE)
    group_pattern = re.compile(r"(group|sub-group|class|sub-class)\b", re.IGNORECASE)

    # Pattern that is a non-semantic ECLASS definition, exclude to not overflow the result file
    exclude_pattern = re.compile(
        r"^Equals to the EphMRA ATC classification code \S+ - to be defined further$",
        re.IGNORECASE
    )

    # Search for non-semantic ECLASS definitions in each segment and save them
    found_definitions = []
    seen_definitions = set()
    for segment in list(range(13, 52)) + [90]:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue

        # Load ECLASS classes from CSV file
        input_path = f"../../data/extracted/eclass-{segment}.csv"
        try:
            database = pd.read_csv(input_path)
            logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
        except Exception as e:
            logger.error(f"Failed to read {input_path}: {e}")
            continue

        # Search for the patterns
        for idx, row in database.iterrows():
            definition = row["definition"]
            if exclude_pattern.search(definition):
                continue
            if group_pattern.search(definition) or no_def_pattern.search(definition):
                if definition not in seen_definitions:
                    logger.info(f"Match: {definition}")
                    seen_definitions.add(definition)
                    found_definitions.append(definition)

    # Save results
    save_path = "potential_non_semantic_definitions.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for definition in found_definitions:
            f.write(definition + "\n")
        f.write("\n")
        f.write("Manual additions:\n")
        f.write("-\n")
        f.write("Equals to the EphMRA ATC classification code XXXX - to be defined further")
