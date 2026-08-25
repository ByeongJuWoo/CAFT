import pandas as pd
import argparse

def csv_to_parquet(input_path, output_path):
    df = pd.read_csv(input_path)
    df.to_parquet(output_path, index=False)
    print("finished")

if __name__ == "__main__":
    # [CC3M] official DreamLIP caption CSV (Image Path=URL + 7 caption columns)
    input_path = '/path/to/data/dreamlip_30m/cc3m_3long_3short_1raw_captions_url.csv'
    # [CC3M] output of step 1: parquet (input to step 2, parquet_to_dreamlip.py)
    output_path = '/path/to/data/dreamlip3m/cc3m_dreamlip.parquet'
    csv_to_parquet(input_path, output_path)
