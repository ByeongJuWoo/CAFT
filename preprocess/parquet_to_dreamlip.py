import os 
import argparse

import pandas as pd
import webdataset as wds
import json
import glob
import subprocess
import tempfile
import shutil

# [CC3M] original cc3m webdataset shards (pixparse/cc3m-wds, train split only)
cc3m_shard_pattern = "/path/to/data/cc3m/cc3m_data/cc3m-train-*.tar"
# [CC3M] output of step 1 (convert_to_parquet.py)
dreamlip_parquet = "/path/to/data/dreamlip3m/cc3m_dreamlip.parquet"
# [CC3M] output of step 2: shards with captions merged into json (input to step 3, presplit_captions.py)
out_dir = "/path/to/data/dreamlip3m/merged"
# [CC3M] temp directory for tar extraction (uses the target drive instead of the default /tmp)
tmp_root = "/path/to/data/dreamlip3m/tmp"
required_keys = [ "raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_root, exist_ok=True)

df_dl = pd.read_parquet(dreamlip_parquet)
mapping = {
    row["Image Path"]: {
        k: v for k, v in row.items() if k != "Image Path"
    }
    for _, row in df_dl.iterrows()
}

print(f"loaded {len(mapping)} dreamlip entries")
shards = sorted(glob.glob(cc3m_shard_pattern))

# valid_json_count = json_count = 0
# for json_dict in mapping.values():
#     json_count +=1
#     if all(k in json_dict for k in required_keys):
#         valid_json_count += 1
# => 100%: 10,010,225

total_json=0
json_with_url = 0
matched=0
unmatched=0

no_url = 0


for shard in shards:
    shard_name = os.path.basename(shard)
    print(f"processing {shard_name}")

    tmpdir = tempfile.mkdtemp(prefix="cc3m_", dir=tmp_root)
    subprocess.check_call(["tar", "-xf", shard, "-C", tmpdir])

    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            total_json = total_json + 1

            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            url = meta.get("url")
            dl_data = mapping.get(url)
            if dl_data is not None:
                matched = matched + 1
                meta.update(dl_data)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
            else:
                unmatched = unmatched + 1
                os.remove(path)
            # with open(path, "r+", encoding="utf-8") as f:
            #     meta = json.load(f)
            #     url = meta.get("url")
            #     dl_data = mapping.get(url)
            #     if dl_data is not None:
            #         matched = matched + 1
            #         meta.update(dl_data)
            #         f.seek(0)
            #         json.dump(meta, f, ensure_ascii=False)
            #         f.truncate()
            #     else:
            #         unmatched = unmatched + 1
           
    out_shard = os.path.join(out_dir, shard_name)
    subprocess.check_call([
        "tar", "-cf", out_shard,"-C", tmpdir, "."
    ])
    print(f"wroted updated shard: {out_shard}")

    shutil.rmtree(tmpdir)
    # break

print("finished")
print(f"matched: {matched}, unmatched: {unmatched}")
print(f"total: {total_json}")
print(f"json with: {json_with_url}")

# for CC3M
# matched: 2823423, unmatched: 82531
# total: 2905954 (97.2%)

# for CC12M
# matched: 10012845, unmatched: 955694
# total: 10968539 (91.3%)