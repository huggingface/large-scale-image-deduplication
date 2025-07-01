#!/usr/bin/env python3
"""
Image Embeddings Counter

This script counts all image embeddings saved in the embeddings-lmms folder
and provides detailed statistics about the datasets.

Usage: python count_embeddings.py
"""

import numpy as np
import os
import glob
from pathlib import Path


def count_embeddings(embeddings_dir="embeddings-lmms"):
    """
    Count all image embeddings in the specified directory.
    
    Args:
        embeddings_dir (str): Path to the directory containing embedding files
        
    Returns:
        tuple: (total_embeddings, dataset_counts, embedding_files_processed)
    """
    # Check if directory exists
    if not os.path.exists(embeddings_dir):
        print(f"Error: Directory '{embeddings_dir}' not found!")
        return 0, [], 0
    
    # Find all embedding files
    embedding_pattern = os.path.join(embeddings_dir, "*_embeddings.npy")
    embedding_files = glob.glob(embedding_pattern)
    embedding_files.sort()
    
    if not embedding_files:
        print(f"No embedding files found in '{embeddings_dir}'")
        return 0, [], 0
    
    total_embeddings = 0
    dataset_counts = []
    errors = []
    
    print(f'Counting image embeddings in {embeddings_dir}/')
    print('=' * 80)
    
    for file in embedding_files:
        try:
            embeddings = np.load(file)
            num_embeddings = embeddings.shape[0]
            embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else "N/A"
            total_embeddings += num_embeddings
            
            # Extract dataset name for cleaner display
            basename = os.path.basename(file)
            dataset_name = basename.replace('lmms-lab-', '').replace('_test_embeddings.npy', '')
            dataset_counts.append((dataset_name, num_embeddings, embedding_dim))
            
            print(f'{dataset_name:45} {num_embeddings:>10,} embeddings (dim: {embedding_dim})')
            
        except Exception as e:
            error_msg = f'Error loading {os.path.basename(file)}: {e}'
            errors.append(error_msg)
            print(error_msg)
    
    return total_embeddings, dataset_counts, len(embedding_files), errors


def print_statistics(total_embeddings, dataset_counts, files_processed, errors):
    """Print detailed statistics about the embeddings."""
    
    print('=' * 80)
    print(f'{"TOTAL":45} {total_embeddings:>10,} embeddings')
    print()
    print(f'Summary:')
    print(f'  • Total image embeddings: {total_embeddings:,}')
    print(f'  • Number of datasets: {len(dataset_counts)}')
    print(f'  • Embedding files processed: {files_processed}')
    
    if errors:
        print(f'  • Errors encountered: {len(errors)}')
        print('\nErrors:')
        for error in errors:
            print(f'    ⚠️  {error}')
    
    if dataset_counts:
        # Show top 15 datasets by number of embeddings
        print('\nTop 15 datasets by number of embeddings:')
        print('-' * 60)
        sorted_datasets = sorted(dataset_counts, key=lambda x: x[1], reverse=True)
        for i, (name, count, dim) in enumerate(sorted_datasets[:15]):
            print(f'{i+1:2}. {name:35} {count:>10,} (dim: {dim})')
        
        if len(sorted_datasets) > 15:
            remaining = len(sorted_datasets) - 15
            remaining_total = sum(count for _, count, _ in sorted_datasets[15:])
            print(f'{"... and " + str(remaining) + " more datasets":38} {remaining_total:>10,}')
        
        # Show size distribution
        print('\nDataset size distribution:')
        print('-' * 40)
        size_ranges = [
            ('Large (>100k)', lambda x: x > 100_000),
            ('Medium (10k-100k)', lambda x: 10_000 <= x <= 100_000),
            ('Small (1k-10k)', lambda x: 1_000 <= x < 10_000),
            ('Tiny (<1k)', lambda x: x < 1_000)
        ]
        
        for range_name, condition in size_ranges:
            datasets_in_range = [d for d in dataset_counts if condition(d[1])]
            count = len(datasets_in_range)
            total_embeddings_in_range = sum(d[1] for d in datasets_in_range)
            if count > 0:
                print(f'{range_name:15} {count:>3} datasets, {total_embeddings_in_range:>10,} embeddings')


def main():
    """Main function to run the embedding counter."""
    print("Image Embeddings Counter")
    print("=" * 50)
    
    # Try to find the embeddings directory
    possible_paths = [
        "embeddings-lmms",
        "./embeddings-lmms", 
        "../embeddings-lmms"
    ]
    
    embeddings_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            embeddings_dir = path
            break
    
    if embeddings_dir is None:
        print("Could not find 'embeddings-lmms' directory!")
        print("Please make sure you're running this script from the correct location.")
        print("Expected directory structure:")
        print("  your-project/")
        print("  ├── count_embeddings.py")
        print("  └── embeddings-lmms/")
        return
    
    print(f"Found embeddings directory: {os.path.abspath(embeddings_dir)}")
    print()
    
    # Count embeddings
    total_embeddings, dataset_counts, files_processed, errors = count_embeddings(embeddings_dir)
    
    if total_embeddings > 0:
        print_statistics(total_embeddings, dataset_counts, files_processed, errors)
    else:
        print("No embeddings found or errors occurred.")


if __name__ == "__main__":
    main() 