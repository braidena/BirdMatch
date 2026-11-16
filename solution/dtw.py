"""
Reference solution: DTW distance (Task 1).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    DTW distance with Euclidean local cost and the standard
    (i-1,j), (i,j-1), (i-1,j-1) steps. O(n*m) time.

    Speed note: all frame-to-frame distances are computed once, in one
    vectorized call, instead of inside the double loop. The loop itself
    then only does list lookups and comparisons.
    """
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    cost = cdist(a, b).tolist()
    n, m = len(cost), len(cost[0])
    inf = float("inf")

    prev = [0.0] + [inf] * m           # row i-1 of D (row 0 is the boundary)
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)          # cur[0] = inf for i > 0
        row = cost[i - 1]
        for j in range(1, m + 1):
            diag, up, left = prev[j - 1], prev[j], cur[j - 1]
            best = diag if (diag < up and diag < left) else (up if up < left else left)
            cur[j] = row[j - 1] + best
        prev = cur
    return prev[m]


def dtw_distance_banded(seq_a: np.ndarray, seq_b: np.ndarray, radius: int) -> float:
    """Sakoe-Chiba banded DTW: only cells with |i - j| <= radius are reachable."""
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    n, m = len(a), len(b)
    if radius < abs(n - m):
        raise ValueError("radius must be >= |n - m| for a finite alignment to exist")
    cost = cdist(a, b).tolist()
    inf = float("inf")

    prev = [0.0] + [inf] * m
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        row = cost[i - 1]
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            diag, up, left = prev[j - 1], prev[j], cur[j - 1]
            best = diag if (diag < up and diag < left) else (up if up < left else left)
            cur[j] = row[j - 1] + best
        prev = cur
    return prev[m]
