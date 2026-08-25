#!/usr/bin/env python3
"""
_claude_convert_to_parquet.py  (step 1)

Low-memory version of convert_to_parquet.py.
- The original loads the whole 5.9GB CSV into memory via pd.read_csv, but
  this machine only has 8GB RAM, so this streams it block-by-block with
  pyarrow (open_csv -> ParquetWriter) instead. The resulting parquet has the
  same schema as the original script's output.
- Captions can contain newlines, so newlines_in_values=True is required.
- Writes to .tmp then finalizes with os.replace (an interrupted run can't
  leave a broken parquet file behind).

Usage:
  python preprocess/_claude_convert_to_parquet.py
"""
import os
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# [CC3M] official DreamLIP caption CSV (Image Path=URL + 7 caption columns)
INPUT_CSV = "/path/to/data/dreamlip_30m/cc3m_3long_3short_1raw_captions_url.csv"
# [CC3M] output of step 1: parquet (input to step 2, _claude_parquet_to_dreamlip.py)
OUTPUT_PARQUET = "/path/to/data/dreamlip3m/cc3m_dreamlip.parquet"


def csv_to_parquet_streaming(input_path, output_path, block_size=64 << 20):
    tmp_path = output_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    read_opts = pacsv.ReadOptions(block_size=block_size)
    parse_opts = pacsv.ParseOptions(newlines_in_values=True)

    writer = None
    total = 0
    with pacsv.open_csv(input_path, read_options=read_opts, parse_options=parse_opts) as reader:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, batch.schema)
            writer.write_batch(batch)
            total += batch.num_rows
            print(f"[progress] {total:,} rows", flush=True)
    if writer is not None:
        writer.close()

    os.replace(tmp_path, output_path)
    print(f"finished: {total:,} rows -> {output_path}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    csv_to_parquet_streaming(INPUT_CSV, OUTPUT_PARQUET)
