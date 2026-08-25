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
cc3m_shard_pattern = "/path/to/data/yfcc15m_tar/*.tar" #"/path/to/data/cc3m/cc3m_data/cc3m-train-*.tar" #"/path/to/data/cc12m_pixparse/*.tar"
dreamlip_parquet = "/path/to/data/yfcc15m_dreamlip.parquet"# "/path/to/data/cc3m_dreamlip.parquet"#"/path/to/data/cc12m_dreamlip.parquet"
out_dir = "/path/to/data/yfcc15m_dreamlip"#"/path/to/data/cc3m_dreamlip" #"/path/to/data/cc12m_dreamlip"
required_keys = [ "raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]
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

    tmpdir = tempfile.mkdtemp(prefix="yfcc15m_")
    subprocess.check_call(["tar", "-xf", shard, "-C", tmpdir])

    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            total_json = total_json + 1

            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            url = meta.get("downloadurl")

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

print("finished")
print(f"matched: {matched}, unmatched: {unmatched}")
print(f"total: {total_json}")
print(f"json with: {json_with_url}")

# finished
# matched: 7699, unmatched: 496
# total: 8195
# json with: 0

# for CC3M
# matched: 2823423, unmatched: 82531 # 2826240
# total: 2905954 (97.2%)

# for CC12M
# matched: 10012845, unmatched: 955694 # 10027008
# total: 10968539 (91.3%)

# for YFCC15M
# matched: 14066050, unmatched: 759183 # 14082048
# total: 14825233 (94.9%)

