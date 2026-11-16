"""
search.py -- TASKS 2 and 3
============================
TASK 2: NaiveSearcher   -- brute-force matching using DTW against every
                            clip in the dataset.
TASK 3: LSHIndex / IndexedSearcher -- approximate nearest-neighbor search
                            that scales to a large dataset by avoiding
                            most of those expensive DTW comparisons.

Read blackbox/audio_features.py's `summarize_features` docstring first --
it explains why the indexed path needs a fixed-length summary vector
instead of the raw (variable-length) MFCC sequence that DTW uses.

You will import your own dtw_distance from dtw.py (Task 1) here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dtw import dtw_distance


@dataclass
class DatasetEntry:
    """One item in the reference dataset."""
    clip_id: str
    label: str                 # species name (ground truth)
    features: np.ndarray       # shape (n_frames, n_mfcc), from extract_features
    summary: np.ndarray        # shape (2 * n_mfcc,), from summarize_features


# ---------------------------------------------------------------------------
# TASK 2: brute-force baseline
# ---------------------------------------------------------------------------

class NaiveSearcher:
    """
    Brute-force matcher: on every query, compute the DTW distance between
    the query and EVERY entry in the dataset, and return the closest
    matches. This is the correctness baseline you will compare the indexed
    searcher against, both for accuracy and for speed.
    """

    def __init__(self, dataset: Sequence[DatasetEntry]):
        """
        Parameters
        ----------
        dataset : sequence of DatasetEntry
            The reference set to search against.
        """
        self.dataset = list(dataset)

    def query(self, query_features: np.ndarray, k: int = 5) -> list[tuple[DatasetEntry, float]]:
        """
        Find the k closest dataset entries to `query_features` by DTW
        distance.

        Parameters
        ----------
        query_features : np.ndarray, shape (n_frames, n_mfcc)
            Feature matrix of the query clip (e.g. from
            `extract_features`), NOT the summary vector.
        k : int
            Number of top matches to return.

        Returns
        -------
        results : list of (DatasetEntry, distance) tuples
            The k closest entries, sorted by ascending DTW distance.
        """
        # TODO: implement me
        raise NotImplementedError("NaiveSearcher.query is not implemented yet")


# ---------------------------------------------------------------------------
# TASK 3: scalable approximate search
# ---------------------------------------------------------------------------

class LSHIndex:
    """
    Locality-Sensitive Hashing index over fixed-length summary vectors,
    using random hyperplane hashing (SimHash-style), which approximates
    nearest-neighbor search under cosine similarity.

    Background
    ----------
    For a summary vector v in R^d, a random hyperplane hash function is
    defined by a random vector r in R^d:

        h_r(v) = 1  if r . v >= 0
                 0  otherwise

    Two vectors that are close together in angle are very likely to land
    on the same side of a random hyperplane (their hash bits agree with
    probability related to the angle between them); vectors that are far
    apart are more likely to disagree. By combining several such bits into
    one hash "signature", and using several independent hash tables (each
    with its own random hyperplanes), you get a structure where vectors
    that hash into the same bucket in ANY table are good candidates for
    being near neighbors -- without ever comparing them to every other
    point.

    You will use this to generate a short candidate list quickly, then
    hand that candidate list to DTW for an accurate final ranking (see
    IndexedSearcher below). This mirrors how real large-scale similarity
    search systems combine a cheap approximate filter with an expensive
    accurate reranker.
    """

    def __init__(self, dim: int, num_tables: int = 4, num_hyperplanes: int = 8, seed: int = 0):
        """
        Parameters
        ----------
        dim : int
            Dimensionality of the summary vectors being indexed.
        num_tables : int
            Number of independent hash tables (L in the LSH literature).
            More tables improve recall at the cost of memory/lookup time.
        num_hyperplanes : int
            Number of random hyperplanes combined into one hash signature
            per table (k in the LSH literature). More hyperplanes make
            each bucket more selective (higher precision, lower recall).
        seed : int
            RNG seed, for reproducible hyperplanes.
        """
        self.dim = dim
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self.rng = np.random.default_rng(seed)
        # TODO: initialize your random hyperplanes and empty hash tables here
        raise NotImplementedError("LSHIndex.__init__ is not implemented yet")

    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        """
        Compute the hash bucket (as an int, e.g. by packing bits) for
        `vector` under hash table `table_idx`.
        """
        # TODO: implement me
        raise NotImplementedError("LSHIndex._hash is not implemented yet")

    def add(self, item_id: str, vector: np.ndarray) -> None:
        """Insert one item (by id) into all hash tables."""
        # TODO: implement me
        raise NotImplementedError("LSHIndex.add is not implemented yet")

    def build(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Convenience: add many items at once. vectors has shape (N, dim)."""
        for item_id, vector in zip(ids, vectors):
            self.add(item_id, vector)

    def candidates(self, query_vector: np.ndarray) -> set[str]:
        """
        Return the union, across all hash tables, of item ids that share
        a bucket with `query_vector`. This is your approximate candidate
        set -- it should typically be much smaller than the full dataset.
        """
        # TODO: implement me
        raise NotImplementedError("LSHIndex.candidates is not implemented yet")


class IndexedSearcher:
    """
    Two-stage searcher: use an LSHIndex to cheaply narrow the dataset down
    to a candidate shortlist, then rerank only that shortlist with exact
    DTW distance (using your NaiveSearcher-style logic, or dtw_distance
    directly) to produce the final top-k.
    """

    def __init__(self, dataset: Sequence[DatasetEntry], num_tables: int = 4, num_hyperplanes: int = 8):
        """
        Build the LSH index over the dataset's summary vectors.

        Parameters
        ----------
        dataset : sequence of DatasetEntry
        num_tables, num_hyperplanes : LSHIndex parameters (see above)
        """
        self.dataset = list(dataset)
        self.by_id = {entry.clip_id: entry for entry in self.dataset}
        # TODO: build self.index = LSHIndex(...) and populate it from
        # self.dataset's summary vectors
        raise NotImplementedError("IndexedSearcher.__init__ is not implemented yet")

    def query(self, query_features: np.ndarray, query_summary: np.ndarray, k: int = 5) -> list[tuple[DatasetEntry, float]]:
        """
        Parameters
        ----------
        query_features : np.ndarray, shape (n_frames, n_mfcc)
            Full feature matrix of the query, used for the DTW rerank step.
        query_summary : np.ndarray, shape (2 * n_mfcc,)
            Fixed-length summary vector of the query, used to look up
            candidates in the LSH index.
        k : int
            Number of top matches to return.

        Returns
        -------
        results : list of (DatasetEntry, distance) tuples
            The (approximate) k closest entries, sorted by ascending DTW
            distance, computed by DTW-reranking only the LSH candidate
            set. If the candidate set has fewer than k items, return as
            many as are available.
        """
        # TODO: implement me
        raise NotImplementedError("IndexedSearcher.query is not implemented yet")
