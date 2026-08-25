import argparse, os, json, gzip, zipfile
from urllib.parse import urlparse
from pathlib import Path

def open_text_auto(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")

def candidate_names(rec):
    ext = (rec.get("ext") or "jpg").lower().lstrip(".")
    exts = [ext]
    if ext == "jpg": exts.append("jpeg")
    if ext == "jpeg": exts.append("jpg")

    cands = []
    k = rec.get("key")
    assert isinstance(k, str) and k 
    for e in exts: cands.append(f"{k.lower()}.{e}")
    return cands

meta_path = "/path/to/data/yfcc15m/snapshots/986e65392adb1f3bdab07c25ed9a23cb83a0b354/metadata/metadata_00a.jsonl.gz"
zip_path = "/path/to/data/yfcc15m/snapshots/986e65392adb1f3bdab07c25ed9a23cb83a0b354/data/00a.zip"
out_path = "/path/to/data/temp/minimal/found.jpg"

with open_text_auto(meta_path) as f:
    for line in f:
        line = line.strip()
        if line:
            rec = json.loads(line)
            print(rec["description_clean"])
            break
cands = candidate_names(rec)
print(cands[0])
with zipfile.ZipFile(zip_path) as zf:
    for n in zf.namelist():
        basename = os.path.basename(n).lower()
        # print(basename)
        if basename == cands[0]:
            print("found!")
            data = zf.read(n)
            with open(out_path, "wb") as f:
                f.write(data)