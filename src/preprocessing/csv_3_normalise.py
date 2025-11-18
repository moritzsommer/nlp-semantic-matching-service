"""Module to normalise ECLASS CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.csv_io import iter_csv_files, load_csv
from utils.logger import LoggerFactory


class Normaliser:
    """Normalise the definition column of ECLASS CSV files."""

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
    def _trim_wrappers(s: str) -> str:
        s = s.strip()
        leading = '([{"\'«“‚‹'
        trailing = ']}"\'»”’›'
        while s and s[0] in leading:
            s = s[1:].lstrip()
        while s and s[-1] in trailing:
            s = s[:-1].rstrip()
        return s

    def _normalise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.id_col, self.name_col, self.def_col])

        # Convert to tuples
        tuples = list(
            df[[self.id_col, self.name_col, self.def_col]]
            .astype(str)
            .itertuples(index=False, name=None)
        )

        # Normalise and return
        rows_out = []
        for id, name, definition in tuples:
            def_norm = self.normalise_definition(definition)
            rows_out.append((id, name, def_norm))
        return pd.DataFrame(rows_out, columns=[self.id_col, self.name_col, self.def_col])

    @staticmethod
    def normalise_definition(s: Optional[str]) -> str:
        """Return a normalised string."""

        if not s:
            return ""

        # Replace space-like characters
        trans = {
            0x00A0: 0x0020,  # NO-BREAK SPACE
            0x202F: 0x0020  # NARROW NO-BREAK SPACE
        }
        s = s.translate(trans)

        s = s.strip()
        s = Normaliser._trim_wrappers(s)
        s = " ".join(s.split())
        s = s.lower()

        return s

    def run(self) -> None:
        """Execute the normaliser across several CSV files."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for infile in iter_csv_files(self.input_dir):
            df = load_csv(infile, self.id_col, self.name_col, self.def_col, self.logger)
            if df is None:
                continue

            df_norm = self._normalise_df(df)

            outfile = self.output_dir / infile.name
            try:
                df_norm.to_csv(outfile, index=False)
                self.logger.info(
                    "Normalised %s -> %s",
                    infile, outfile
                )
            except Exception as e:
                self.logger.error("Failed to write %s: %s", outfile, e)


if __name__ == "__main__":
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Running ECLASS definition normaliser...")

    normaliser = Normaliser(
        input_dir="../../data/extracted-classes/2-filtered-classes",
        output_dir="../../data/extracted-classes/3-normalised-classes",
        logger=logger,
    )
    normaliser.run()

    normaliser = Normaliser(
        input_dir="../../data/extracted-properties/2-filtered-properties",
        output_dir="../../data/extracted-properties/3-normalised-properties",
        logger=logger,
    )
    normaliser.run()
