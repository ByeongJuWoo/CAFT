import os 
import argparse

import pandas as pd
import webdataset as wds
import json
import glob
import subprocess
import tempfile
import shutil

cc3m_dir = "/path/to/data/cc3m/cc3m_data"
cc3m_shard_pattern = "/path/to/data/cc3m/cc3m_data/cc3m-train-*.tar"
dreamlip_parquet = "/path/to/data/cc3m_dreamlip.parquet"
out_dir = "/path/to/data/cc3m_dreamlip"
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

    for fn in os.listdir(tmpdir):   
        if not fn.endswith(".json"):
            continue
        total_json +=1
        key = os.path.splitext(fn)[0]
        json_path = os.path.join(tmpdir, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        url = meta.get("url")
        dl_data = mapping.get(url)

        if dl_data is None:
            unmatched += 1
            continue
        matched +=1
        for col, val in dl_data.items():
            out_path = os.path.join(tmpdir, f"{key}.{col}")
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(str(val))
           
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
