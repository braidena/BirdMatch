#!/usr/bin/env python3
"""
package_dataset.py
==================
Turn the downloaded recordings (from fetch_xeno_canto.py) into the dataset
zip that students receive:

    birdmatch_dataset.zip
      manifest.csv          id, label, recordist, split (reference|query), feature path, n_frames
      metadata.csv          full Xeno-canto metadata (recordist, license, url, ...)
      ATTRIBUTION.md        credits + license notice
      dataset_info.json     the processing settings used
      feature_stats.npz     mean/std used to standardize features (for new query clips)
      features/<label>/XC<id>.npy        float32, shape (n_frames, d)
      summaries.npz         ids + fixed-length summary vectors (for the LSH task)
      audio/<label>/XC<id>.<ext>         only with --include-audio

Processing (so DTW stays affordable): each recording is cut to its loudest
--max-seconds stretch, converted to the chosen feature --variant (see
audio_features.VARIANTS), and every --pool consecutive frames are averaged.
Features are then standardized per dimension using statistics from the
REFERENCE set only. Clips shorter than --min-frames after pooling are dropped.

Held-out queries: ~--query-frac of each species' recordings are marked
split=query and must NEVER be inserted into the search index. The split is
by RECORDIST, so near-duplicate takes from one session cannot leak across it.

Usage
-----
    python package_dataset.py --raw ../data/raw_audio --variant mfcc --out ../birdmatch_dataset.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blackbox"))
from audio_features import (  # noqa: E402
    DEFAULT_MAX_SECONDS, DEFAULT_POOL, VARIANTS, apply_standardizer,
    fit_standardizer, raw_variant_features, summarize_features,
)

ATTRIBUTION = """# Attribution and licenses

All recordings in this dataset come from Xeno-canto (https://xeno-canto.org).
Each recording remains the property of its recordist and is distributed under
the Creative Commons license listed for it in `metadata.csv` (columns
`recordist`, `license`, `url`). The feature files are derived from those
recordings and carry the same license terms.

Use is limited to this course project. If you redistribute any part of this
dataset or the audio, keep this file and `metadata.csv` with it, and follow
the license of each individual recording (many are non-commercial and
share-alike).
"""


def split_by_recordist(recs: list[dict], query_frac: float, rng: random.Random) -> set[str]:
    """
    Return the set of recording ids to hold out as queries for one species.
    Whole recordist groups are assigned together. At least one group always
    stays in the reference set; a species with a single recordist gets no
    queries, since there is no clean way to split it.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r.get("recordist", "") or f"unknown-{r['id']}"].append(r)
    if len(groups) < 2:
        return set()
    target = max(1, round(len(recs) * query_frac))
    order = list(groups.values())
    rng.shuffle(order)
    held, count = set(), 0
    for g in order[:-1]:                 # never hold out every group
        if count >= target:
            break
        held.update(r["id"] for r in g)
        count += len(g)
    return held


def decode_one(args):
    """Worker: decode one file, return {variant: pooled raw features} or None."""
    path, variants, max_seconds, pool = args
    try:
        return raw_variant_features(path, variants, max_seconds, pool)
    except Exception as e:                              # undecodable / empty file
        return f"{type(e).__name__}: {e}"


def load_recordings(raw: Path, variants, max_seconds, pool, jobs, min_frames):
    """Decode every recording once. Returns (rows_kept, raw_feats_by_id, n_skipped)."""
    with open(raw / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    work = [(str(raw / r["path"]), tuple(variants), max_seconds, pool) for r in rows]
    if jobs > 1:
        with Pool(jobs) as p:
            results = p.map(decode_one, work, chunksize=8)
    else:
        results = [decode_one(w) for w in work]
    kept, feats, skipped = [], {}, 0
    for r, res in zip(rows, results):
        if isinstance(res, str):
            print(f"skip XC{r['id']}: {res}")
            skipped += 1
        elif next(iter(res.values())).shape[0] < min_frames:
            print(f"skip XC{r['id']}: shorter than {min_frames} frames")
            skipped += 1
        else:
            kept.append(r)
            feats[r["id"]] = res
    return kept, feats, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="../data/raw_audio")
    ap.add_argument("--out", default="../birdmatch_dataset.zip")
    ap.add_argument("--variant", default="mfcc", choices=VARIANTS)
    ap.add_argument("--query-frac", type=float, default=0.2)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--min-frames", type=int, default=20, help="drop clips shorter than this after pooling")
    ap.add_argument("--jobs", type=int, default=2, help="parallel decoding processes")
    ap.add_argument("--include-audio", action="store_true", help="also put the raw audio in the zip (large)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = Path(args.raw)
    if not (raw / "metadata.csv").exists():
        print(f"ERROR: {raw / 'metadata.csv'} not found. Run fetch_xeno_canto.py first.", file=sys.stderr)
        return 2

    rows, raw_feats, skipped = load_recordings(
        raw, [args.variant], args.max_seconds, args.pool, args.jobs, args.min_frames)
    if not rows:
        print("ERROR: no usable recordings.", file=sys.stderr)
        return 1

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(args.seed)
    held: set[str] = set()
    for label, recs in sorted(by_label.items()):
        recs.sort(key=lambda r: int(r["id"]))
        h = split_by_recordist(recs, args.query_frac, rng)
        if not h:
            print(f"warning: {label} has a single recordist; no held-out queries for it")
        held |= h

    ref_ids = [r["id"] for r in rows if r["id"] not in held]
    stats = fit_standardizer(raw_feats[i][args.variant] for i in ref_ids)

    feats_dir = raw.parent / "features"
    manifest, ids, summaries = [], [], []
    for r in rows:
        f = apply_standardizer(raw_feats[r["id"]][args.variant], stats)
        label = r["label"]
        fpath = feats_dir / label / f"XC{r['id']}.npy"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(fpath, f)
        ids.append(f"XC{r['id']}")
        summaries.append(summarize_features(f.astype(np.float64)))
        manifest.append({
            "id": f"XC{r['id']}", "label": label, "recordist": r.get("recordist", ""),
            "split": "query" if r["id"] in held else "reference",
            "feature_path": f"features/{label}/XC{r['id']}.npy",
            "n_frames": f.shape[0],
            "audio_path": f"audio/{r['path']}" if args.include_audio else "",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = feats_dir.parent
    with open(base / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    np.savez_compressed(base / "summaries.npz", ids=np.array(ids), summaries=np.vstack(summaries))
    np.savez(base / "feature_stats.npz", mean=stats[0], std=stats[1])
    frames = [m["n_frames"] for m in manifest]
    info = {
        "variant": args.variant, "dim": int(stats[0].shape[0]), "max_seconds": args.max_seconds,
        "pool": args.pool, "sample_rate": 22050, "hop_length": 256,
        "seconds_per_frame": 256 * args.pool / 22050,
        "n_clips": len(manifest), "n_species": len({m["label"] for m in manifest}),
        "frames_min": int(min(frames)), "frames_mean": float(np.mean(frames)), "frames_max": int(max(frames)),
        "standardized_with": "reference-set statistics (feature_stats.npz)",
        "query_split": "by recordist", "seed": args.seed,
    }
    (base / "dataset_info.json").write_text(json.dumps(info, indent=2))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("manifest.csv", "summaries.npz", "feature_stats.npz", "dataset_info.json"):
            z.write(base / name, name)
        z.write(raw / "metadata.csv", "metadata.csv")
        z.writestr("ATTRIBUTION.md", ATTRIBUTION)
        for m in manifest:
            z.write(base / m["feature_path"], m["feature_path"])
            if args.include_audio:
                z.write(raw / Path(m["audio_path"]).relative_to("audio"), m["audio_path"])

    n_q = sum(m["split"] == "query" for m in manifest)
    print(f"\nWrote {out} ({out.stat().st_size/1e6:.1f} MB): {len(manifest)} clips, "
          f"{n_q} held-out queries, {len(manifest)-n_q} reference, {skipped} skipped.")
    print(f"Variant '{args.variant}', dim {info['dim']}. "
          f"Frames per clip: min {info['frames_min']}, mean {info['frames_mean']:.0f}, max {info['frames_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
