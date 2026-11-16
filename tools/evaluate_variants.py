#!/usr/bin/env python3
"""
evaluate_variants.py
====================
Compare feature recipes on YOUR downloaded audio, so we pick the one that
actually separates species before building the assignment around it.

For each variant it reports, on held-out queries (split by recordist):
  * summary 1-NN accuracy  -- nearest neighbor on the fixed-length summary
                              vector (what LSH will see); all queries, instant
  * DTW 1-NN accuracy      -- brute-force DTW on a stratified sample of queries
  * DTW 5-NN vote accuracy -- majority species among the 5 closest by DTW
Chance is 1 / number_of_species (5% for 20 species).

Usage (from the tools/ folder):
    python evaluate_variants.py --raw ../data/raw_audio --sample 40 --jobs 4

Takes a few minutes: decoding the audio once, then DTW on the sample for
each variant. Use --variants mfcc pcen to test only some.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "blackbox"))
sys.path.insert(0, str(HERE.parent / "solution"))
sys.path.insert(0, str(HERE))
from audio_features import (  # noqa: E402
    DEFAULT_MAX_SECONDS, DEFAULT_POOL, VARIANTS, apply_standardizer,
    fit_standardizer, summarize_features,
)
from dtw import dtw_distance  # noqa: E402
from package_dataset import load_recordings, split_by_recordist  # noqa: E402

_REF = None


def _init(ref):
    global _REF
    _REF = ref


def _dtw_row(q):
    return [dtw_distance(q, r) for r in _REF]


def evaluate(variant, rows, raw_feats, held, labels, sample, jobs, seed):
    ref_rows = [r for r in rows if r["id"] not in held]
    qry_rows = [r for r in rows if r["id"] in held]
    stats = fit_standardizer(raw_feats[r["id"]][variant] for r in ref_rows)
    F = {r["id"]: apply_standardizer(raw_feats[r["id"]][variant], stats) for r in rows}
    ref_lab = np.array([r["label"] for r in ref_rows])

    # --- summary-vector 1-NN (all queries)
    Sr = np.vstack([summarize_features(F[r["id"]].astype(np.float64)) for r in ref_rows])
    Sq = np.vstack([summarize_features(F[r["id"]].astype(np.float64)) for r in qry_rows])
    d = ((Sq[:, None, :] - Sr[None, :, :]) ** 2).sum(-1)
    sum_acc = float(np.mean(ref_lab[d.argmin(1)] == np.array([r["label"] for r in qry_rows])))

    # --- DTW on a stratified sample of queries (round-robin over species)
    rng = random.Random(seed)
    per = defaultdict(list)
    for r in qry_rows:
        per[r["label"]].append(r)
    for v in per.values():
        rng.shuffle(v)
    sub = []
    while len(sub) < min(sample, len(qry_rows)):
        for lab in sorted(per):
            if per[lab] and len(sub) < sample:
                sub.append(per[lab].pop())
    t0 = time.time()
    ref_feats = [F[r["id"]] for r in ref_rows]
    with Pool(jobs, initializer=_init, initargs=(ref_feats,)) as p:
        D = np.array(p.map(_dtw_row, [F[r["id"]] for r in sub], chunksize=2))
    secs = time.time() - t0
    qlab = np.array([r["label"] for r in sub])
    order = np.argsort(D, axis=1)
    top1 = float(np.mean(ref_lab[order[:, 0]] == qlab))
    vote5 = float(np.mean([collections.Counter(ref_lab[order[i, :5]]).most_common(1)[0][0] == qlab[i]
                           for i in range(len(sub))]))
    return {"variant": variant, "dim": int(stats[0].shape[0]), "n_queries_total": len(qry_rows),
            "n_queries_dtw": len(sub), "n_reference": len(ref_rows),
            "summary_1nn": round(sum_acc, 3), "dtw_1nn": round(top1, 3), "dtw_5nn_vote": round(vote5, 3),
            "dtw_seconds": round(secs), "ms_per_pair": round(1000 * secs * jobs / (len(sub) * len(ref_rows)), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="../data/raw_audio")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--sample", type=int, default=40, help="queries used for the (slow) DTW evaluation")
    ap.add_argument("--query-frac", type=float, default=0.2)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="variant_results.json")
    args = ap.parse_args()

    raw = Path(args.raw)
    print(f"Decoding audio once for variants {args.variants} ...")
    rows, raw_feats, _ = load_recordings(raw, args.variants, args.max_seconds, args.pool, args.jobs, args.min_frames)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(args.seed)
    held = set()
    for lab, recs in sorted(by_label.items()):
        recs.sort(key=lambda r: int(r["id"]))
        held |= split_by_recordist(recs, args.query_frac, rng)
    labels = sorted(by_label)
    print(f"{len(rows)} clips, {len(labels)} species, {len(held)} held-out queries "
          f"(chance = {1/len(labels):.3f})\n")

    results = []
    for v in args.variants:
        print(f"evaluating {v} ...", flush=True)
        res = evaluate(v, rows, raw_feats, held, labels, args.sample, args.jobs, args.seed)
        results.append(res)
        print(f"   summary 1-NN {res['summary_1nn']:.3f} | DTW 1-NN {res['dtw_1nn']:.3f} | "
              f"DTW 5-NN vote {res['dtw_5nn_vote']:.3f} | {res['ms_per_pair']} ms/pair")

    print("\nvariant        dim  summary-1NN  DTW-1NN  DTW-5NN-vote")
    for r in results:
        print(f"{r['variant']:<14}{r['dim']:>3}  {r['summary_1nn']:>10.3f}  {r['dtw_1nn']:>7.3f}  {r['dtw_5nn_vote']:>11.3f}")
    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\n(saved {args.json_out}; paste the table above back to Claude)")


if __name__ == "__main__":
    main()
