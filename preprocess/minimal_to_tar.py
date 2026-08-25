import argparse, os, re, json, gzip, zipfile
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import webdataset as wds

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

def candidates_from_key(key: str, ext: "jpg"):
    return [f"{key.lower()}.{ext}" for ext in ["jpg", "jpeg", "png", "webp"]]

SNAPSHOT_ROOT = Path("/path/to/data/yfcc15m/snapshots/986e65392adb1f3bdab07c25ed9a23cb83a0b354")
OUT_TAR_PATH = "/path/to/data/yfcc15m_tar/minimal.tar"
data_dir = SNAPSHOT_ROOT / "data"
meta_dir = SNAPSHOT_ROOT / "metadata"
zips = sorted(data_dir.glob("*.zip"))[: 3]

missing_key = 0
with wds.TarWriter(OUT_TAR_PATH) as sink:
    for zpath in tqdm(zips, desc="zips"):
        shard = zpath.stem
        meta_path = meta_dir / f"metadata_{shard}.jsonl.gz"
        if not meta_path.exists():
            print(f"skip {zpath.name} (missing {meta_path.name})")
            continue
        with zipfile.ZipFile(zpath, "r") as zf:
            stem_map = index_zip_by_key(zf)
            for rec in tqdm(iter_jsonl(meta_path), desc=f"meta {shard}", leave=False):
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
                    "json": json.dumps(rec, ensure_ascii=False).encode("utf-8"),
                    "txt": rec["raw_caption"]
                }
                sink.write(sample)
print(missing_key)

# tar test code here
ds = (wds.WebDataset(OUT_TAR_PATH)
        .decode("pilrgb") 
        .rename(image="jpg", txt="json")
        .to_tuple("image", "txt"))
for i, (img, meta) in enumerate(ds):
    img.save(f"/path/to/data/yfcc15m_tar/{i}.jpg", "JPEG")
    print(i, img.size, meta.get("raw_caption"))
    if i >= 5:
        break



