import os
import os.path as osp
import tarfile
from tqdm import tqdm
import json

tar_paths = [
    '/path/to/data/cc12m_dreamlip/00000.tar',
    '/path/to/data/cc3m_dreamlip/cc3m-train-0016.tar',
    '/path/to/data/cc3m_dreamlip/cc3m-train-0017.tar',
]

required_keys = [ "raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]


def check(path, dreamlip_path='/path/to/data/cc12m_dreamlip/', verbose=True):
    jpg_count = txt_count = json_count = 0
    paired_count = 0
    valid_json_count = 0
    grouped = {}

    with tarfile.open(path, 'r') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            base, ext = osp.splitext(osp.basename(m.name))
            ext = ext.lower()[1:]

            if ext == 'jpg': jpg_count +=1
            if ext == 'txt': txt_count +=1
            if ext == 'json': json_count +=1

            info = grouped.setdefault(base, {})[ext] = (path, m)

    for base, info in grouped.items():
        if 'jpg' in info and 'json' in info:
            paired_count += 1
    
    for info in grouped.values():
        if 'jpg' not in info:
            continue
        path, m = info['json']
        raw = tarfile.open(path, 'r').extractfile(m).read()
        data = json.loads(raw)
        if all(k in data for k in required_keys):
            valid_json_count += 1
    
    if verbose:
        print(f"#jpg: {jpg_count}")
        print(f"#txt: {txt_count}")
        print(f"#json: {json_count}")
        print(f"#paired: {paired_count}")
        print(f"#valid_json: {valid_json_count}")

    if dreamlip_path:
        pass


    return jpg_count, txt_count, json_count, paired_count, valid_json_count

jpg_count, txt_count, json_count, paired_count, valid_json_count = check(tar_paths[0])

