#!/usr/bin/env python3
"""
convert_hf_zip_to_wds.py

Converts a Hugging Face dataset snapshot (e.g. snapshots/<commit>/data/*.zip)
into WebDataset (.tar) shards.

- Streams images out of the ZIPs into .tar without unpacking them to disk
- If metadata/*.jsonl.gz exists, matches it to samples by filename/path and writes .json metadata alongside each sample
- --maxcount controls samples per shard; if shards already exist, continues numbering from the next index (resume)
- --verify checks image decode integrity (Pillow)

Example usage (Linux):
  python convert_hf_zip_to_wds.py \
    --snapshot "/path/to/data/hub/datasets--dalle-mini--YFCC100M_OpenAI_subset/snapshots/<commit>/" \
    --out "/path/to/data/wds/yfcc_subset" \
    --maxcount 10000 --verify

Verify (read back):
  python - << 'PY'
import webdataset as wds
ds = (wds.WebDataset("/path/to/data/wds/yfcc_subset/{000000..000999}.tar")
        .decode("pil").to_tuple("jpg;jpeg;png;webp","json;txt"))
for i,(img,meta) in enumerate(ds):
    if i<3: print(img.size, type(meta))
    else: break
print("OK")
PY
"""
import argparse
import gzip
import io
import json
import re
import sys
import tarfile
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

try:
    import webdataset as wds
except ImportError:
    print("Please: pip install webdataset", file=sys.stderr); sys.exit(1)

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTS = {".json", ".txt"}

def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS

def key_ext(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in (".jpg", ".jpeg"): return "jpg"
    if ext == ".png":            return "png"
    if ext == ".webp":           return "webp"
    return "bin"

def sha1key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def build_meta_index(snapshot: Path, caption_key: Optional[str]) -> Dict[str, dict]:
    """Read metadata/*.jsonl.gz and build an index keyed by basename and relative path"""
    meta_dir = snapshot / "metadata"
    idx: Dict[str, dict] = {}
    if not meta_dir.exists():
        return idx
    for jl in sorted(meta_dir.glob("*.jsonl.gz")):
        try:
            with gzip.open(jl, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    # common file-key candidates
                    fname = (j.get("file_name") or j.get("filename") or
                             j.get("path") or j.get("relpath") or
                             j.get("image") or j.get("img"))
                    if not fname:
                        continue
                    base = Path(fname).name
                    idx.setdefault(base, j)
                    norm = str(Path(fname).as_posix())
                    idx.setdefault(norm, j)
                    # optional: if caption_key is set, cast it to a string
                    if caption_key and caption_key in j and j[caption_key] is not None:
                        j[caption_key] = str(j[caption_key])
        except Exception:
            continue
    return idx

def find_zip_files(snapshot: Path):
    data_dir = snapshot / "data"
    zips = []
    if data_dir.exists():
        zips += sorted(data_dir.glob("*.zip"))
    if not zips:
        zips += sorted(snapshot.glob("**/*.zip"))
    if not zips:
        raise FileNotFoundError("No ZIP files found under snapshot; check --snapshot path.")
    return zips

def iter_zip_images(snapshot: Path, meta_index: Dict[str, dict],
                    caption_key: Optional[str]) -> Iterator[Tuple[str, str, bytes, Optional[bytes], Optional[bytes]]]:
    """
    Yields image bytes from the ZIPs along with their matching metadata.
    Returns: (key, extkey, img_bytes, meta_json_bytes, caption_txt_bytes)
    """
    for zp in find_zip_files(snapshot):
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    name = zi.filename
                    if not is_image(name):
                        continue
                    try:
                        with zf.open(zi, "r") as fh:
                            img_bytes = fh.read()
                        extkey  = key_ext(name)
                        base    = Path(name).name
                        meta    = meta_index.get(base) or meta_index.get(str(Path(name).as_posix()))
                        meta_b  = json.dumps(meta, ensure_ascii=False).encode("utf-8") if meta else None
                        cap_b   = None
                        if meta and caption_key and caption_key in meta and meta[caption_key] is not None:
                            cap_b = str(meta[caption_key]).encode("utf-8")
                        key = sha1key(f"{zp.name}:{name}")
                        yield key, extkey, img_bytes, meta_b, cap_b
                    except Exception:
                        continue
        except Exception:
            # skip the whole zip if it's corrupted
            continue

class RollingTarWriter:
    """Simple writer that auto-continues shard numbering (000000.tar, 000001.tar, ...)"""
    def __init__(self, out_dir: Path, maxcount: int):
        self.out_dir = out_dir
        self.maxcount = maxcount
        self.idx = self._next_index()
        self.count_in_shard = 0
        self.tw: Optional[wds.TarWriter] = None
        self._open_new()

    def _next_index(self) -> int:
        pat = re.compile(r"^(\d{6})\.tar$")
        existing = [int(m.group(1)) for p in self.out_dir.glob("*.tar")
                    if (m:=pat.match(p.name))]
        return (max(existing)+1) if existing else 0

    def _open_new(self):
        if self.tw is not None:
            self.tw.close()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self.out_dir / f"{self.idx:06d}.tar"
        while shard_path.exists():
            self.idx += 1
            shard_path = self.out_dir / f"{self.idx:06d}.tar"
        self.tw = wds.TarWriter(str(shard_path))
        self.count_in_shard = 0

    def write(self, sample: dict):
        self.tw.write(sample)
        self.count_in_shard += 1
        if self.count_in_shard >= self.maxcount:
            self.idx += 1
            self._open_new()

    def close(self):
        if self.tw is not None:
            self.tw.close()
            self.tw = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="path to snapshots/<commit>/ (containing data/*.zip)")
    ap.add_argument("--out",      required=True, help="output folder for WDS shards")
    ap.add_argument("--maxcount", type=int, default=10000, help="samples per shard")
    ap.add_argument("--verify",   action="store_true", help="verify image integrity with Pillow (slower)")
    ap.add_argument("--caption-key", type=str, default=None, help="metadata key to save as the caption .txt (e.g. text/caption)")
    args = ap.parse_args()

    snapshot = Path(args.snapshot)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_index = build_meta_index(snapshot, args.caption_key)
    print(f"[info] metadata indexed: {len(meta_index)} keys")

    if args.verify:
        try:
            from PIL import Image
            def verify_img(b: bytes):
                Image.open(io.BytesIO(b)).verify()
        except Exception:
            print("[warn] Pillow not installed/failed -> ignoring --verify", file=sys.stderr)
            args.verify = False

    writer = RollingTarWriter(out_dir, args.maxcount)
    n = 0
    for key, extkey, img_b, meta_b, cap_b in iter_zip_images(snapshot, meta_index, args.caption_key):
        if args.verify:
            try:
                from PIL import Image
                Image.open(io.BytesIO(img_b)).verify()
            except Exception:
                continue
        sample = {"__key__": key, extkey: img_b}
        if meta_b: sample["json"] = meta_b
        if cap_b:  sample["txt"]  = cap_b
        writer.write(sample)
        n += 1
        if n % 1000 == 0:
            print(f"[progress] {n} samples")

    writer.close()
    print(f"[ok] written samples: {n}")
    print(f"[ok] shards at: {out_dir}")

if __name__ == "__main__":
    main()
