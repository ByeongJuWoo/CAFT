import argparse, os, re, json, gzip, zipfile
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import webdataset as wds
from typing import Dict, Tuple, List, Optional
import multiprocessing as mp

def json_dump(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def iter_jsonl(meta_path: Path):
    if str(meta_path).endswith(".gz"):
        f =  gzip.open(meta_path, "rt", encoding="utf-8")
    else:
        f = open(pmeta_path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def index_zip_by_key(zf: zipfile.ZipFile):
    m = {}
    for name in zf.namelist():
        base = os.path.basename(name)
        key = os.path.splitext(base)[0].lower()
        if key in m:
            raise RuntimeError(f"Duplicated stem in zip: {stem} ({m[key]} vs {name})")
        m[key] = name
    return m

# def worker_run(
#     worker_id: int,
#     zips: List,
#     meta_dir: Path,
#     out_dir: Path,
#     prefix: str,
#     max_size_bytes: int,
#     hearbeat_q: mp.queues.Queue,
#     progress_every: int = 2000,
# ):
#     wout = out_dir / f"w{worker_id:02d}"
#     wout.mkdir(parents=True, exists_ok=True)
#     pattern = str(wout / f"{prefix}-w{worker_id:02d}-%06d.tar")

total_written = 0
total_seen = 0
zip_done = 0
missing_key = 0

SNAPSHOT_ROOT = Path("/path/to/data/yfcc15m/snapshots/986e65392adb1f3bdab07c25ed9a23cb83a0b354")
OUT_TAR_PATH = Path("/path/to/data/yfcc15m_tar/")
OUT_TAR_PATH.mkdir(parents=True, exist_ok=True)
data_dir = SNAPSHOT_ROOT / "data"
meta_dir = SNAPSHOT_ROOT / "metadata"
zips = sorted(data_dir.glob("*.zip"))
prefix = "yfcc15m"

pattern = str(OUT_TAR_PATH / f"{prefix}-%06d.tar")
max_size_bytes = int(1024 ** 3)

with wds.ShardWriter(pattern, maxsize=max_size_bytes, start_shard=0) as sink:
    for zpath in tqdm(zips, desc="zips"):
        shard = zpath.stem
        meta_path = meta_dir / f"metadata_{shard}.jsonl.gz"
        if not meta_path.exists():
            htqdm.write((f"[WARN] skip {zpath.name} (missing {meta_path.name})"))
            continue

        with zipfile.ZipFile(zpath, "r") as zf:
            stem_map = index_zip_by_key(zf)
            local_written = 0
            local_seen = 0
            
            for rec in tqdm(iter_jsonl(meta_path), desc=f"meta {shard}", leave=False):
                local_seen += 1
                key = rec.get("key")
                if not key:
                    continue
                key = str(key).lower()
                rec["raw_caption"] = rec.get("description_clean", "")
                
                full_name = stem_map.get(key)
                if not full_name:
                    missing_key = missing_key + 1
                    continue

                img_bytes = zf.read(full_name)
                sample = {
                    "__key__": f"{shard}-{key}",
                    "jpg": img_bytes,
                    "json": json_dump(rec),
                    "txt": rec["raw_caption"]
                }
                sink.write(sample)
                local_written += 1
            total_written += local_written
            total_seen += local_seen
            tqdm.write(f"[OK] {zpath.name}: written={local_written}, seen={local_seen}")

print(f"[DONE] written={total_written}, seen={total_seen}")
print(missing_key)





