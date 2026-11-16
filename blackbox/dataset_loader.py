"""
dataset_loader.py
=================
PROVIDED MODULE. Loads the unzipped birdmatch dataset (see the handout for
the folder layout) into Python objects.

    reference, queries = load_dataset("path/to/unzipped/dataset")

* `reference` is the collection you build your search structures over.
* `queries` are held-out "unknown" songs. They are never part of the
  reference set; the true species of each is in `query.label` so you can
  score your answers, but your search code must not look at it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetEntry:
    """One recording."""
    clip_id: str
    label: str                 # species (ground truth)
    features: np.ndarray       # shape (n_frames, n_mfcc), float32
    summary: np.ndarray        # fixed-length vector, from summarize_features


def load_dataset(root: str) -> tuple[list[DatasetEntry], list[DatasetEntry]]:
    """Return (reference_entries, query_entries) from an unzipped dataset folder."""
    root_path = Path(root)
    with open(root_path / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    z = np.load(root_path / "summaries.npz")
    summary_by_id = {str(i): s for i, s in zip(z["ids"], z["summaries"])}

    reference, queries = [], []
    for r in rows:
        entry = DatasetEntry(
            clip_id=r["id"],
            label=r["label"],
            features=np.load(root_path / r["feature_path"]),
            summary=summary_by_id[r["id"]],
        )
        (queries if r["split"] == "query" else reference).append(entry)
    return reference, queries
