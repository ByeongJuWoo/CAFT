import os 
import argparse

import pandas as pd
import webdataset as wds
import json
import glob
import subprocess
import tempfile
import shutil

# cc3m_dir = "/path/to/data/cc3m/cc3m_data"
cc3m_shard_pattern = "/path/to/data/cc12m/data/*.tar"
dreamlip_parquet = "/path/to/data/cc12m_dreamlip.parquet"
out_dir = "/path/to/data/cc12m_dreamlip"
os.makedirs(out_dir, exist_ok=True)

df_dl = pd.read_parquet(dreamlip_parquet)
mapping = {
    row["Image Path"]: {
        k: v for k, v in row.items() if k != "Image Path"
    }
    for _, row in df_dl.iterrows()
}

print(f"loaded {len(mapping)} dreamlip entries")
shards = sorted(glob.glob(cc3m_shard_pattern))

total_json=0
matched=0
unmatched=0

for shard in shards:
    shard_name = os.path.basename(shard)
    print(f"processing {shard_name}")

    tmpdir = tempfile.mkdtemp(prefix="cc3m_")
    subprocess.check_call(["tar", "-xf", shard, "-C", tmpdir])

    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            total_json = total_json + 1
            with open(path, "r+", encoding="utf-8") as f:
                meta = json.load(f)
                url = meta.get("url")
                dl_data = mapping.get(url)
                if dl_data is not None:
                    matched = matched + 1
                    meta.update(dl_data)
                    f.seek(0)
                    json.dump(meta, f, ensure_ascii=False)
                    f.truncate()
                else:
                    unmatched = unmatched + 1
           
    out_shard = os.path.join(out_dir, shard_name)
    subprocess.check_call([
        "tar", "-cf", out_shard,"-C", tmpdir, "."
    ])
    print(f"wroted updated shard: {out_shard}")

    shutil.rmtree(tmpdir)

print("finished")
print(f"matched: {matched}, unmatched: {unmatched}")
print(f"total: {total_json}")


# matched: 2823423, unmatched: 82531
#total: 2905954
