"""Module to filter out non-semantic definitions from ECLASS CSV files in bulk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from preprocessing.filter_non_semantic import (
    filter_definitions_missing, filter_definitions_missing_suffix,
    filter_definitions_structural)
from utils.csv_io import iter_csv_files, load_csv
from utils.logger import LoggerFactory

MISSING_SET = frozenset(filter_definitions_missing)
MISSING_SUFFIXES = tuple(filter_definitions_missing_suffix)
STRUCTURAL_SET = frozenset(filter_definitions_structural)


class Filter:
    """Filter out non-semantic definitions."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        id_col: str = "id",
        name_col: str = "preferred-name",
        def_col: str = "definition",
        logger: Optional[logging.Logger] = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.id_col = id_col
        self.name_col = name_col
        self.def_col = def_col

        self.logger = logger or LoggerFactory.get_logger(__name__)

    @staticmethod
    def _is_semantic(definition: str) -> bool:
        if not definition:
            return False
        if definition in MISSING_SET:
            return False
        if definition.endswith(MISSING_SUFFIXES):
            return False
        if definition in STRUCTURAL_SET:
            return False
        return True

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.id_col, self.name_col, self.def_col])

        # Convert to tuples
        tuples = list(
            df[[self.id_col, self.name_col, self.def_col]]
            .astype(str)
            .itertuples(index=False, name=None)
        )

        # Filter and return
        filtered = self.filter_semantic_pairs(tuples)
        return pd.DataFrame(filtered, columns=[self.id_col, self.name_col, self.def_col])

    @staticmethod
    def filter_semantic_pairs(rows: Iterable[Tuple[str, str, str]]):
        """Filter definitions, their names and ids by excluding entries with missing and placeholder definitions."""

        out = []
        for id, name, definition in rows:
            if Filter._is_semantic(definition):
                out.append((id, name, definition))
        return out

    def run(self) -> None:
        """Execute the filter across several CSV files."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for infile in iter_csv_files(self.input_dir):
            df = load_csv(infile, self.id_col, self.name_col, self.def_col, self.logger)
            if df is None:
                continue

            df_filtered = self._filter_df(df)

            outfile = self.output_dir / infile.name
            try:
                df_filtered.to_csv(outfile, index=False)
                self.logger.info(
                    "Filtered %s -> %s (kept rows: %d)",
                    infile, outfile, len(df_filtered)
                )
            except Exception as e:
                self.logger.error("Failed to write %s: %s", outfile, e)


if __name__ == "__main__":
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Running ECLASS definition filter...")

    filter = Filter(
        input_dir="../../data/extracted-classes/1-original-classes",
        output_dir="../../data/extracted-classes/2-filtered-classes",
        logger=logger,
    )
    filter.run()

    filter = Filter(
        input_dir="../../data/extracted-properties/1-original-properties",
        output_dir="../../data/extracted-properties/2-filtered-properties",
        logger=logger,
    )
    filter.run()
