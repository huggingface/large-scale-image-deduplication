import argparse
import numpy as np
import os
import glob
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from compute_embeddings import compute_embeddings

class Timer:
    """Context manager for timing operations."""
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.elapsed = elapsed

def find_duplicates_against_precomputed(new_embeddings_file, new_image_ids_file, precomputed_dir, batch_size=5000, threshold=0.90):
    """Find duplicates by comparing new embeddings against all precomputed embeddings."""
    # Load new dataset embeddings
    new_embeddings = np.load(new_embeddings_file)
    new_image_ids = np.load(new_image_ids_file)
    
    print(f"Loaded {len(new_embeddings)} new embeddings")
    
    # Get all precomputed embedding files
    precomputed_files = glob.glob(os.path.join(precomputed_dir, '*_embeddings.npy'))
    print(f"Found {len(precomputed_files)} precomputed embedding files")
    
    duplicate_indices = set()
    duplicate_details = []
    loading_time = 0.0
    similarity_time = 0.0
    
    # Compare against each precomputed file
    for i, precomputed_file in enumerate(precomputed_files):
        print(f"Comparing against precomputed file {i+1}/{len(precomputed_files)}: {os.path.basename(precomputed_file)}")
        
        # Time loading of precomputed embeddings
        with Timer("Loading precomputed embeddings") as load_timer:
            precomputed_embeddings = np.load(precomputed_file)
        loading_time += load_timer.elapsed
        
        # Process new embeddings in batches against this precomputed file
        for batch_start in range(0, len(new_embeddings), batch_size):
            batch_end = min(batch_start + batch_size, len(new_embeddings))
            batch_embeddings = new_embeddings[batch_start:batch_end]
            
            # Time similarity computation
            with Timer("Computing similarities") as sim_timer:
                similarities = cosine_similarity(batch_embeddings, precomputed_embeddings)
                # Find duplicates above threshold
                batch_indices, precomputed_indices = np.where(similarities >= threshold)
            similarity_time += sim_timer.elapsed
            
            # Record duplicates
            for batch_idx, precomputed_idx in zip(batch_indices, precomputed_indices):
                global_idx = batch_start + batch_idx
                duplicate_indices.add(global_idx)
                
                duplicate_details.append({
                    'new_idx': int(global_idx),
                    'new_image_id': int(new_image_ids[global_idx]),
                    'source_file': os.path.basename(precomputed_file),
                    'source_idx': int(precomputed_idx),
                    'similarity': float(similarities[batch_idx, precomputed_idx])
                })
    
    return sorted(duplicate_indices), duplicate_details, loading_time, similarity_time

def deduplicate_dataset(dataset_name, name=None, split='test', threshold=0.90, precomputed_dir='embeddings-lmms', output_dir='duplicates'):
    """Deduplicate a dataset against precomputed embeddings with timing information."""
    
    with Timer("Total execution time") as total_timer:
        # Step 1: Compute embeddings for new dataset
        print("Step 1: Computing embeddings for new dataset...")
        with Timer("Computing embeddings") as embedding_timer:
            compute_embeddings(dataset_name, name=name, split=split, output_dir='embeddings-tmp')
        
        # Step 2: Find duplicate files
        sanitized_dataset_name = dataset_name.replace('/', '-')
        if name:
            sanitized_name = name.replace('/', '-')
            filename_base = f'{sanitized_dataset_name}_{sanitized_name}_{split}'
        else:
            filename_base = f'{sanitized_dataset_name}_{split}'
        
        new_embeddings_file = f'embeddings-tmp/{filename_base}_embeddings.npy'
        new_image_ids_file = f'embeddings-tmp/{filename_base}_image_ids.npy'
        
        # Step 3: Find duplicates
        print("Step 2: Finding duplicates against precomputed embeddings...")
        with Timer("Duplicate detection") as duplicate_timer:
            duplicate_indices, duplicate_details, loading_time, similarity_time = find_duplicates_against_precomputed(
                new_embeddings_file, new_image_ids_file, precomputed_dir, threshold=threshold
            )
        
    # Step 4: Save results (outside timer context to access elapsed times)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/duplicates_{filename_base}.json'
    
    total_images = int(len(np.load(new_image_ids_file)))
    results = {
        'dataset_name': dataset_name,
        'name': name,
        'split': split,
        'threshold': float(threshold),
        'total_images': total_images,
        'timing': {
            'total_time': total_timer.elapsed,
            'embedding_computation': embedding_timer.elapsed,
            'duplicate_detection': duplicate_timer.elapsed,
            'loading_precomputed': loading_time,
            'similarity_search': similarity_time
        },
        'duplicate_count': len(duplicate_indices),
        'duplicate_indices': [int(idx) for idx in duplicate_indices],
        'duplicate_details': duplicate_details,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print timing summary
    print(f"\nTiming Summary:")
    print(f"Total execution: {total_timer.elapsed:.5f} seconds")
    print(f"Embedding computation: {embedding_timer.elapsed:.5f} seconds ({embedding_timer.elapsed/total_timer.elapsed*100:.1f}%)")
    print(f"Duplicate detection: {duplicate_timer.elapsed:.5f} seconds ({duplicate_timer.elapsed/total_timer.elapsed*100:.1f}%)")
    print(f"  - Loading precomputed: {loading_time:.5f} seconds ({loading_time/total_timer.elapsed*100:.1f}%)")
    print(f"  - Similarity search: {similarity_time:.5f} seconds ({similarity_time/total_timer.elapsed*100:.1f}%)")
    print(f"\nFound {len(duplicate_indices)} duplicate images out of {total_images}")
    print(f"Duplicate results saved to: {output_file}")
    
    return duplicate_indices, duplicate_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate HuggingFace dataset against precomputed embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to process")
    parser.add_argument("--name", type=str, default=None, help="Dataset (subset) name")
    parser.add_argument("--precomputed_dir", type=str, help="Directory containing precomputed embeddings")
    parser.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold for duplicate detection")
    parser.add_argument("--output_dir", type=str, default="duplicates", help="Output directory for duplicate results")

    args = parser.parse_args()
    deduplicate_dataset(args.dataset, args.name, args.split, args.threshold, args.precomputed_dir, args.output_dir) 