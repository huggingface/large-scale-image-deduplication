#!/usr/bin/env python3

from lmms_datasets import LMMS_NAMES
from compute_embeddings import compute_embeddings

def main():
    for dataset_name, subset_names in LMMS_NAMES.items():
        if subset_names:  # Dataset has subsets
            for subset_name in subset_names:
                print(f"Processing {dataset_name} with subset {subset_name}")
                compute_embeddings(dataset_name, name=subset_name)
        else:  # Dataset has no subsets
            print(f"Processing {dataset_name}")
            compute_embeddings(dataset_name)

if __name__ == "__main__":
    main() 