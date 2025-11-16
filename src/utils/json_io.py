"""Module providing helper functions to load and save JSON files."""

import json


def save_json(data, path):
    """Save data as JSON."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """Load data from JSON."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
