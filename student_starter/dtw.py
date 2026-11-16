"""
dtw.py -- TASK 1
=================
Implement Dynamic Time Warping (DTW) distance between two MFCC feature
matrices (as produced by blackbox/audio_features.py).

Background
----------
Two recordings of the same bird call are almost never the same length in
frames: the bird might sing a little faster, a little slower, or pause
briefly. If you just compared frame 1 to frame 1, frame 2 to frame 2, ...,
a small timing shift would make two calls from the SAME species look very
different, and could make calls from DIFFERENT species look deceptively
similar.

DTW solves this by finding the lowest-cost way to "stretch" and "compress"
the two sequences in time so that similar-sounding parts line up, then
reports the total alignment cost as the distance between the sequences.
It is a classic dynamic programming problem, closely related to edit
distance / longest common subsequence.

Task
----
Implement `dtw_distance(seq_a, seq_b)` below.

Let seq_a have shape (n, d) and seq_b have shape (m, d) (n and m may
differ; d, the feature dimension, is always the same for both since they
come from the same feature extractor).

Define a local cost between frame i of seq_a and frame j of seq_b as the
Euclidean distance between those two feature vectors:

    cost(i, j) = || seq_a[i] - seq_b[j] ||_2

Define the cumulative cost matrix D of shape (n+1, m+1) by:

    D[0, 0] = 0
    D[i, 0] = inf   for i > 0
    D[0, j] = inf   for j > 0
    D[i, j] = cost(i-1, j-1) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
              for i in [1..n], j in [1..m]

The DTW distance between seq_a and seq_b is D[n, m].

(This is one standard formulation of DTW; you may use any textbook-
equivalent recurrence, but it must (a) run in O(n*m) time and space in
its basic form, and (b) reduce to the same result on the provided test
cases.)

Requirements
------------
- Do not call any external DTW library (no `dtw-python`, no `fastdtw`,
  no `librosa.sequence.dtw`, etc.) -- the point of this task is to
  implement the dynamic program yourself. numpy for array operations is
  fine and encouraged.
- `dtw_distance` must run correctly for n != m (different-length inputs).
- See tests/test_dtw.py for the exact behavior your implementation will
  be checked against.

Extra credit (optional)
------------------------
`dtw_distance_banded(seq_a, seq_b, radius)` restricts the alignment path
to a Sakoe-Chiba band of the given radius around the diagonal (i.e. only
compute D[i, j] where |i - j| <= radius, treating everything outside the
band as infinite cost). This turns DTW from O(n*m) into O(n*radius) time,
at the cost of being unable to find alignments that stray far from the
diagonal. Implement this if you want to explore the runtime/accuracy
trade-off in your benchmark report.
"""

from __future__ import annotations

import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute the DTW distance between two feature sequences.

    Parameters
    ----------
    seq_a : np.ndarray, shape (n, d)
    seq_b : np.ndarray, shape (m, d)

    Returns
    -------
    distance : float
        The DTW alignment cost between seq_a and seq_b, as defined above.
    """
    # TODO: implement me
    raise NotImplementedError("dtw_distance is not implemented yet")


def dtw_distance_banded(seq_a: np.ndarray, seq_b: np.ndarray, radius: int) -> float:
    """
    (Optional, extra credit) Sakoe-Chiba banded DTW: same as dtw_distance,
    but only cells with |i - j| <= radius are considered reachable; all
    others are treated as cost = inf.

    Parameters
    ----------
    seq_a : np.ndarray, shape (n, d)
    seq_b : np.ndarray, shape (m, d)
    radius : int
        Band half-width. radius >= abs(n - m) is required for a valid
        (finite-cost) alignment to exist.

    Returns
    -------
    distance : float
    """
    # TODO (optional): implement me
    raise NotImplementedError("dtw_distance_banded is not implemented yet")
