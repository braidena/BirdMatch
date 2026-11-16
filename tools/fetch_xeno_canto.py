#!/usr/bin/env python3
"""
fetch_xeno_canto.py
===================
Download a labeled bird-recording dataset from Xeno-canto (API v3).

Run this on YOUR OWN machine/network (it needs to reach xeno-canto.org).

Setup
-----
1. Find your personal API key on your Xeno-canto account page (you need a
   registered account with a verified email).
2. Put it in an environment variable. Do NOT paste it into a file that
   goes into git or into the assignment zip:

       export XC_API_KEY="your-key-here"          # macOS / Linux
       setx XC_API_KEY "your-key-here"            # Windows (new terminal after)

Usage
-----
    python fetch_xeno_canto.py --species-file species.txt --per-species 60 \
        --out ../data/raw_audio

Output
------
    <out>/<Genus_species>/XC<id>.<ext>   one audio file per recording
    <out>/metadata.csv                   one row per downloaded recording
                                         (species label, recordist, license, ...)

The script is resumable: re-running it skips recordings already listed in
metadata.csv and files already on disk.

Notes on being a good API citizen
---------------------------------
* The API only accepts tagged queries, so every request is built from
  sp:"genus species" (plus grp:birds, quality and length filters).
* Requests are spaced out by --delay seconds. Keep it >= 0.5.
* "Download everything" (the whole archive, on the order of a million
  recordings) is not what this script is for; use --per-species to cap
  each species. Check Xeno-canto's terms of use before large downloads.
* Recordings are under per-recording Creative Commons licenses. By default
  this script SKIPS NoDerivatives (-nd) licenses, because the assignment
  trims/resamples audio. Pass --allow-nd to include them.
* Restricted species come back redacted (no file URL); they are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "BirdMatch-course-project/1.0 (educational; respects API rate guidance)"

METADATA_FIELDS = [
    "id", "species", "english_name", "label", "recordist", "country",
    "type", "quality", "length", "license", "url", "background_species",
    "path",
]


def mask(text: str, key: str) -> str:
    """Never print the API key."""
    return text.replace(key, "***KEY***") if key else text


def http_get(url: str, key: str, retries: int = 3, timeout: int = 60):
    """GET with retries/backoff. Returns (bytes, content_type)."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (400, 401, 403, 404):  # not worth retrying
                break
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed: {mask(str(last_err), key)}")


def build_query(species: str, quality: str | None, max_seconds: int | None,
                sound_type: str | None) -> str:
    parts = [f'sp:"{species}"', "grp:birds"]
    if quality:                       # e.g. "A" -> q:A ; "B" -> q:">C" (A or B)
        parts.append('q:">C"' if quality.upper() == "AB" else f"q:{quality.upper()}")
    if max_seconds:
        parts.append(f'len:"<{max_seconds}"')
    if sound_type:
        parts.append(f"type:{sound_type}")
    return " ".join(parts)


def search_species(base_url: str, key: str, query: str, max_pages: int, delay: float):
    """Yield recording objects for a query, following pagination."""
    page = 1
    while page <= max_pages:
        params = {"query": query, "key": key, "per_page": 500, "page": page}
        url = base_url + "?" + urllib.parse.urlencode(params)
        body, _ = http_get(url, key)
        data = json.loads(body.decode("utf-8"))
        if "error" in data:
            raise RuntimeError(f"API error: {data['error']}")
        for rec in data.get("recordings", []):
            yield rec
        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(delay)


def usable(rec: dict, allow_nd: bool, clean_only: bool) -> bool:
    file_url = rec.get("file", "")
    if not isinstance(file_url, str) or not file_url.startswith("http"):
        return False                                   # redacted / restricted species
    if rec.get("status", "identified") != "identified":
        return False                                   # uncertain labels hurt accuracy
    lic = (rec.get("lic") or "").lower()
    if not allow_nd and "-nd" in lic:
        return False
    if clean_only and rec.get("also"):
        return False                                   # background species present
    return True


def slug(species: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", species.strip()).strip("_")


def load_done(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    with open(meta_path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-file", default="species.txt",
                    help='text file, one "Genus species" per line (# for comments)')
    ap.add_argument("--per-species", type=int, default=60)
    ap.add_argument("--out", default="../data/raw_audio")
    ap.add_argument("--quality", default="AB", help="A, B, C, ... or AB (A or B). Default AB")
    ap.add_argument("--max-seconds", type=int, default=60, help="skip recordings longer than this")
    ap.add_argument("--type", dest="sound_type", default=None, help='e.g. song, call (default: any)')
    ap.add_argument("--clean-only", action="store_true", help="skip recordings with background species")
    ap.add_argument("--allow-nd", action="store_true", help="include NoDerivatives licenses")
    ap.add_argument("--max-pages", type=int, default=4, help="pages of 500 results to scan per species")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between API/download requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="override (used for offline testing)")
    args = ap.parse_args()

    key = os.environ.get("XC_API_KEY", "").strip()
    if not key:
        print("ERROR: set the XC_API_KEY environment variable to your Xeno-canto API key.", file=sys.stderr)
        return 2

    species_list = [ln.strip() for ln in Path(args.species_file).read_text().splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    done = load_done(meta_path)
    new_file = not meta_path.exists()
    rng = random.Random(args.seed)

    consecutive_failures = 0
    with open(meta_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=METADATA_FIELDS)
        if new_file:
            writer.writeheader()

        for species in species_list:
            query = build_query(species, args.quality, args.max_seconds, args.sound_type)
            print(f"\n== {species}  (query: {query})")
            try:
                pool = [r for r in search_species(args.base_url, key, query, args.max_pages, args.delay)
                        if usable(r, args.allow_nd, args.clean_only)]
            except RuntimeError as e:
                print(f"  search failed: {e}")
                continue
            rng.shuffle(pool)                    # spread across recordists/countries
            already = sum(1 for r in pool if r["id"] in done)
            print(f"  {len(pool)} usable recordings found ({already} already downloaded)")

            got = already
            for rec in pool:
                if got >= args.per_species:
                    break
                if rec["id"] in done:
                    continue
                name_hint = rec.get("file-name", "") or ""
                ext = Path(name_hint).suffix.lower() or ".mp3"
                dest = out / slug(species) / f"XC{rec['id']}{ext}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    body, ctype = http_get(rec["file"], key)
                    if "html" in ctype.lower() or "json" in ctype.lower():
                        raise RuntimeError(
                            f"got {ctype!r} instead of audio (a bot-check page?). "
                            "Try again later, or download this file in a browser.")
                    dest.write_bytes(body)
                except RuntimeError as e:
                    consecutive_failures += 1
                    print(f"  XC{rec['id']}: {e}")
                    if consecutive_failures >= 5:
                        print("Too many consecutive failures; stopping. Re-run later to resume.")
                        return 1
                    continue
                consecutive_failures = 0
                writer.writerow({
                    "id": rec["id"],
                    "species": f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
                    "english_name": rec.get("en", ""),
                    "label": slug(species),
                    "recordist": rec.get("rec", ""),
                    "country": rec.get("cnt", ""),
                    "type": rec.get("type", ""),
                    "quality": rec.get("q", ""),
                    "length": rec.get("length", ""),
                    "license": rec.get("lic", ""),
                    "url": rec.get("url", ""),
                    "background_species": ";".join(rec.get("also") or []),
                    "path": str(dest.relative_to(out)),
                })
                mf.flush()
                done.add(rec["id"])
                got += 1
                print(f"  [{got}/{args.per_species}] XC{rec['id']} -> {dest.relative_to(out)}")
                time.sleep(args.delay)

    print(f"\nDone. Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
