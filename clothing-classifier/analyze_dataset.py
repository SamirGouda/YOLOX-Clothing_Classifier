import argparse
import sys
import os
from pathlib import Path
import pandas as pd
from dataset import ClothingSmallDataset

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        help='path to images',
                        default='clothing-dataset-small',
    )
    try:
        params = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    return params

def create_csv_from_dataset(input_dir: Path, file: str):
    dataset = ClothingSmallDataset(f'{input_dir}/{file}')
    dataset.to_csv(f'{input_dir}/{file}.csv')

def do_analysis(input_dir):
    for file in ['train', 'test', 'validation']:
        if not os.path.exists(os.path.join(input_dir, f'{file}.csv')):
            create_csv_from_dataset(input_dir, file)
        df = pd.read_csv(f'{input_dir}/{file}.csv')
        counts = df['label'].value_counts()
        print(counts)

def main(input_dir: Path):
    do_analysis(input_dir)

if __name__ == "__main__":
    print(' '.join(sys.argv))   # print command line for logging
    params = parser_arguments()
    main(params.input_dir)