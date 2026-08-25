#!/usr/bin/env python3
"""
_claude_parquet_to_dreamlip.py  (combined steps 2+3, option A)

Runs parquet_to_dreamlip.py (merging caption json) and presplit_captions.py
(splitting captions into sentences) in a single pass. Produces only the
final training shards, without an intermediate merged/ output.

Changes vs. the original:
- RAM 8GB workaround: instead of an in-memory dict for the 2.8M-row caption
  table, builds a SQLite (url -> pre-split json) disk index for lookups.
  The index is built once and reused on subsequent runs.
- Caption sentence-splitting (split_caption) is applied up front, at index-build
  time -> shard processing only needs to merge json (no separate presplit step).
- resume: skips a shard if the output already exists. Writes go to .tmp then
  os.replace, so an interrupted run can't leave a broken shard mistaken for a
  finished one.
- --limit N: process only the first N shards then exit (for a quick test
  before a full run).

The original cc3m data is read-only and is never modified or deleted.

Usage:
  # test (1 shard):
  python preprocess/_claude_parquet_to_dreamlip.py --limit 1
  # full run:
  python preprocess/_claude_parquet_to_dreamlip.py

Reference expected numbers (from the original pipeline run): matched 2,823,423 / total 2,905,954 (97.2%)
"""
import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile

import pyarrow.parquet as pq

# [CC3M] original cc3m webdataset shards (pixparse/cc3m-wds, train split only, read-only)
CC3M_SHARD_PATTERN = "/path/to/data/cc3m/cc3m_data/cc3m-train-*.tar"
# [CC3M] output of step 1 (_claude_convert_to_parquet.py)
DREAMLIP_PARQUET = "/path/to/data/dreamlip3m/cc3m_dreamlip.parquet"
# [CC3M] final output: training shards with captions merged in as sentence lists
OUT_DIR = "/path/to/data/dreamlip3m/shards"
# [CC3M] SQLite index for url -> caption lookup (built once, reused afterward)
INDEX_DB = "/path/to/data/dreamlip3m/cc3m_dreamlip_index.sqlite"
# [CC3M] temp directory for tar extraction (uses the target drive instead of the default /tmp)
TMP_ROOT = "/path/to/data/dreamlip3m/tmp"

CAPTION_KEYS = [
    "raw_caption",
    "shortIB_captions", "longIB_captions",
    "shortSV_captions", "longSV_captions",
    "shortLLA_captions", "longLLA_captions",
]


def split_caption(text):
    # same logic as presplit_captions.py
    texts = re.split(r'\n|</s>|[.]', text)
    subcap = []
    for text_prompt in texts:
        text_prompt = text_prompt.strip()
        if len(text_prompt) >= 3:  # original was: !=0. however, i found some sentence like '1.'
            subcap.append(text_prompt)
    return subcap


def build_index(parquet_path, db_path):
    if os.path.exists(db_path):
        print(f"[index] reuse existing: {db_path}")
        return
    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    con = sqlite3.connect(tmp_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE cap (url TEXT PRIMARY KEY, data TEXT)")

    pf = pq.ParquetFile(parquet_path)
    total = 0
    for batch in pf.iter_batches(batch_size=20000):
        cols = batch.to_pydict()
        urls = cols["Image Path"]
        rows = []
        for i, url in enumerate(urls):
            entry = {}
            for k in CAPTION_KEYS:
                v = cols[k][i]
                entry[k] = split_caption(v) if isinstance(v, str) else []
            rows.append((url, json.dumps(entry, ensure_ascii=False)))
        con.executemany("INSERT OR IGNORE INTO cap VALUES (?, ?)", rows)
        total += len(rows)
        print(f"[index] {total:,} rows", flush=True)

    con.commit()
    con.close()
    os.replace(tmp_path, db_path)
    print(f"[index] built: {total:,} rows -> {db_path}")


def cleanup_leftovers():
    """Remove leftover temp output from an interrupted previous run (assumes no concurrent runs)"""
    for p in glob.glob(os.path.join(OUT_DIR, "*.tmp")):
        print(f"[cleanup] removing incomplete shard: {p}")
        os.remove(p)
    for d in glob.glob(os.path.join(TMP_ROOT, "cc3m_*")):
        print(f"[cleanup] removing orphan tmpdir: {d}")
        shutil.rmtree(d, ignore_errors=True)


def process_shards(limit=None):
    con = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True)
    lookup = con.execute

    shards = sorted(glob.glob(CC3M_SHARD_PATTERN))
    if limit is not None:
        shards = shards[:limit]
    print(f"[info] {len(shards)} shards to process")

    total_json = matched = unmatched = 0

    for shard in shards:
        shard_name = os.path.basename(shard)
        out_shard = os.path.join(OUT_DIR, shard_name)
        if os.path.exists(out_shard):
            print(f"[skip] {shard_name} (already done)")
            continue
        print(f"processing {shard_name}", flush=True)

        tmpdir = tempfile.mkdtemp(prefix="cc3m_", dir=TMP_ROOT)
        try:
            subprocess.check_call(["tar", "-xf", shard, "-C", tmpdir])

            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    if not fn.endswith(".json"):
                        continue
                    path = os.path.join(root, fn)
                    total_json += 1

                    with open(path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    url = meta.get("url")
                    row = lookup("SELECT data FROM cap WHERE url = ?", (url,)).fetchone() if url else None
                    if row is not None:
                        matched += 1
                        meta.update(json.loads(row[0]))
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False)
                    else:
                        # same as the original pipeline: remove the json for samples that fail to match
                        # (filtered out by the training loader's filter_no_caption_or_no_image_json)
                        unmatched += 1
                        os.remove(path)

            tmp_out = out_shard + ".tmp"
            subprocess.check_call(["tar", "-cf", tmp_out, "-C", tmpdir, "."])
            os.replace(tmp_out, out_shard)
            print(f"wrote shard: {out_shard}", flush=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    con.close()
    print("finished")
    print(f"matched: {matched}, unmatched: {unmatched}")
    print(f"total: {total_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N shards (for testing)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)
    cleanup_leftovers()
    build_index(DREAMLIP_PARQUET, INDEX_DB)
    process_shards(limit=args.limit)
