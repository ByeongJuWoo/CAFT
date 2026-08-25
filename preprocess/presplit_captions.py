import argparse
import os
import tarfile
import re
import json
from multiprocessing import Pool
from tqdm import tqdm
from io import BytesIO

CAPTION_KEYS = [
    "raw_caption",
    "shortIB_captions", "longIB_captions",
    "shortSV_captions", "longSV_captions",
    "shortLLA_captions", "longLLA_captions",
]

# [CC3M] output of step 3 (final): training shards with captions pre-split into sentence lists
OUTPUT_DIR = "/path/to/data/dreamlip3m/shards"

def split_caption(text):
    texts = re.split(r'\n|</s>|[.]', text)
    subcap = []
    for text_prompt in texts:
        text_prompt = text_prompt.strip()
        if len(text_prompt) >= 3: # original was: !=0. however, i found some sentence like '1.'
            subcap.append(text_prompt)
    return subcap

def process_tar(tar_path):
    basename = os.path.basename(tar_path)
    tmp_tar_path = os.path.join(OUTPUT_DIR, basename + ".tmp")
    out_tar_path = os.path.join(OUTPUT_DIR, basename)
    try:
        with tarfile.open(tar_path, 'r') as in_tar, tarfile.open(tmp_tar_path, 'w') as out_tar:
            for member in in_tar.getmembers():
                if member.isdir():
                    continue
                file_bytes = in_tar.extractfile(member).read()

                if member.name.endswith('.json'):
                    json_obj = json.loads(file_bytes.decode('utf-8'))
                    for k in CAPTION_KEYS:
                        if k in json_obj and isinstance(json_obj[k], str):
                            json_obj[k] = split_caption(json_obj[k])
                    file_bytes = json.dumps(json_obj).encode('utf-8')

                info = tarfile.TarInfo(name=member.name)
                info.size = len(file_bytes)
                out_tar.addfile(info, BytesIO(file_bytes))
        
        os.replace(tmp_tar_path, out_tar_path)
        return (out_tar_path, "success")

    except Exception as e:
        return (tar_path, f"failed {e}")

if __name__ == "__main__":
    # [CC3M] directory holding the output of step 2 (parquet_to_dreamlip.py)
    shards_dir = "/path/to/data/dreamlip3m/merged"
    num_process = 8  # limited due to external SSD I/O bottleneck (originally 24)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shard_paths = [
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.endswith('.tar')
    ]
    with Pool(processes=num_process) as pool:
        results = list(tqdm(pool.imap_unordered(process_tar, shard_paths), total=len(shard_paths)))
    for tar, status in results:
        print(f"{tar}: {status}")