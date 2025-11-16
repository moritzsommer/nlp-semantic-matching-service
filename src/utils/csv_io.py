"""Module providing helper functions to load CSV files."""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def iter_csv_files(input_dir: Path) -> Iterable[Path]:
    """Find all CSV files in a directory."""

    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".csv":
            yield f


def load_csv(file: Path, id_col: str, name_col: str, def_col: str, logger) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame."""

    try:
        df = pd.read_csv(file, sep=",")
        missing = {id_col, name_col, def_col} - set(df.columns)
        if missing:
            logger.error(f"Missing required column: {missing}")
            return None
        return df
    except Exception as e:
        logger.error(f"Failed to read file: {file}, Error: {e}")
        return None
