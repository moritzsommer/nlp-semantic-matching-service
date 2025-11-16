"""Module providing helper functions to find an IRDI in an ECLASS CSV file."""

from pathlib import Path
from typing import Optional

import pandas as pd


def lookup_definition(irdi: str, dir_path: str) -> Optional[str]:
    """Look up the definition for a specific IRDI in a combined ECLASS CSV file."""

    dir_path = Path(dir_path)
    csv_path = dir_path / "eclass-0.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, sep=",", dtype=str)
    except Exception:
        return None

    if "id" not in df.columns or "definition" not in df.columns:
        return None
    row = df[df["id"] == irdi]
    return row.iloc[0]["definition"] if not row.empty else None


def lookup_name(irdi: str, dir_path: str) -> Optional[str]:
    """Look up the name for a specific IRDI in a combined ECLASS CSV file."""

    dir_path = Path(dir_path)
    csv_path = dir_path / "eclass-0.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, sep=",", dtype=str)
    except Exception:
        return None

    if "id" not in df.columns or "preferred-name" not in df.columns:
        return None
    row = df[df["id"] == irdi]
    return row.iloc[0]["preferred-name"] if not row.empty else None
