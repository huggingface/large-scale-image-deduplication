import torch
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
import time
import math

# Constants
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "models/sscd_disc_mixup.torchscript.pt"

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
        # print(f"{self.operation_name}: {elapsed:.5f} seconds")
        self.elapsed = elapsed

def load_model(device):
    """Load and prepare the SSCD model."""
    print("Loading model...")
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    return model.to(device)

def create_transform():
    """Create the image transformation pipeline."""
    normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    return transforms.Compose([
        transforms.Resize([320, 320]), #transforms.Resize(288),
        #transforms.CenterCrop(150),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #transforms.RandomAffine(degrees=10, translate=[0.1, 0.2], scale=[0.8, 0.9], shear=5),
        transforms.ToTensor(),
        normalize,
    ])

def load_precomputed_data(embeddings_file, image_ids_file):
    """Load precomputed embeddings and image IDs."""
    print("Loading precomputed embeddings...")
    embeddings = np.load(embeddings_file)
    image_ids = np.load(image_ids_file)
    return embeddings, image_ids

def process_query_images(image_paths, transform, device):
    """Load and transform query images into a batch tensor."""
    print(f"Loading & Processing {len(image_paths)} query image(s)...")
    image_tensors = []
    
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        image_tensors.append(tensor)
    
    return torch.cat(image_tensors, dim=0).to(device), image_tensors

def compute_embeddings(model, batch_tensor):
    """Compute embeddings for the batch of query images."""
    print(f"Computing embeddings for {len(batch_tensor)} image(s)...")
    with torch.no_grad():
        return model(batch_tensor).cpu().numpy()

def find_top_similarities(query_embeddings, precomputed_embeddings, top_k):
    """Find top-k most similar images for all queries efficiently."""
    print(f"Computing similarities for {len(query_embeddings)} image(s)...")
    similarities = cosine_similarity(query_embeddings, precomputed_embeddings)
    
    # Get top-k indices and similarities in one operation
    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]
    top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
    
    return top_indices, top_similarities

def print_results(image_paths, top_indices, top_similarities, image_ids, top_k):
    """Print similarity results for each query image."""
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\nQuery image {i+1}: {img_path}")
        print(f"Top {top_k} most similar images:")
        
        for j, (idx, sim, img_id) in enumerate(
            zip(top_indices[i], top_similarities[i], image_ids[top_indices[i]])
        ):
            print(f"  {j+1}. Image ID: {img_id}, Similarity: {sim:.4f}")
        
        results.append((top_indices[i], top_similarities[i], image_ids[top_indices[i]]))
    
    return results

def print_timing_summary(total_timer, loading_timer, model_timer, similarity_timer, num_images):
    """Print a comprehensive timing summary."""
    total_time = total_timer.elapsed
    
    print(f"\nTiming Summary:")
    print(f"Total execution: {total_time:.5f} seconds")
    print(f"Loading: {loading_timer.elapsed:.5f} seconds ({loading_timer.elapsed/total_time*100:.1f}%)")
    print(f"Model inference: {model_timer.elapsed:.5f} seconds ({model_timer.elapsed/total_time*100:.1f}%)")
    print(f"Similarity search: {similarity_timer.elapsed:.5f} seconds ({similarity_timer.elapsed/total_time*100:.1f}%) | Average per image: {similarity_timer.elapsed/num_images:.5f} seconds")

def denormalize_image(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def calculate_grid_size(total_images):
    """Calculate optimal grid dimensions for displaying images."""
    # Try to make the grid as square as possible
    cols = math.ceil(math.sqrt(total_images))
    rows = math.ceil(total_images / cols)
    
    # Ensure minimum dimensions for readability
    cols = max(cols, 2)
    rows = max(rows, 2)
    
    return rows, cols

def visualize_results(query_tensor, dataset, top_image_ids, top_similarities, output_file, top_k):
    """Create and save visualization of query image and top similar images."""
    total_images = top_k + 1  # +1 for query image
    rows, cols = calculate_grid_size(total_images)
    
    # Adjust figure size based on grid dimensions
    fig_width = cols * 5
    fig_height = rows * 5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is always 2D for consistent indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Display query image in the first position
    query_img = denormalize_image(query_tensor).permute(1, 2, 0).numpy()
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image\n(After Transform)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Generate positions for similar images (skip the first position)
    positions = []
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:  # Skip query image position
                continue
            positions.append((row, col))
            if len(positions) >= top_k:  # Stop when we have enough positions
                break
        if len(positions) >= top_k:
            break
    
    # Display similar images
    for i, (pos, img_id, sim) in enumerate(zip(positions, top_image_ids, top_similarities)):
        row, col = pos
        try:
            dataset_item = dataset[int(img_id)]
            
            # Handle both 'image' and 'images' keys
            image_data = None
            if 'images' in dataset_item:
                image_data = dataset_item['images']
            elif 'image' in dataset_item:
                image_data = dataset_item['image']
            else:
                raise KeyError("No 'image' or 'images' key found")
            
            # Handle both single images and lists of images
            if isinstance(image_data, list):
                # If it's a list, use the first image
                if len(image_data) > 0 and image_data[0] is not None:
                    similar_img = image_data[0].convert('RGB')
                else:
                    raise ValueError("Empty or invalid image list")
            else:
                # If it's a single image
                if image_data is not None:
                    similar_img = image_data.convert('RGB')
                else:
                    raise ValueError("Image data is None")
            
            axes[row, col].imshow(similar_img)
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f"Error loading\nimage {img_id}\n{str(e)[:50]}", 
                          ha='center', va='center', transform=axes[row, col].transAxes)
        
        axes[row, col].set_title(f"#{i+1} Similar\nID: {img_id}\nSim: {sim:.5f}", fontsize=10)
        axes[row, col].axis('off')
    
    # Hide any unused subplot positions
    for row in range(rows):
        for col in range(cols):
            if (row == 0 and col == 0):  # Query image position
                continue
            position_index = row * cols + col - 1  # -1 because we skip (0,0)
            if position_index >= top_k:  # Hide unused positions
                axes[row, col].axis('off')
                axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")

def find_similar_images(image_paths, embeddings_file, image_ids_file, dataset_name=None, name=None, split='val', top_k=5):
    """Find the most similar images to one or more query images using precomputed embeddings."""
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with Timer("Total execution time") as total_timer:
        # Load data and model
        with Timer("Loading time") as loading_timer:
            precomputed_embeddings, image_ids = load_precomputed_data(embeddings_file, image_ids_file)
            model = load_model(device)
            transform = create_transform()
            batch_tensor, image_tensors = process_query_images(image_paths, transform, device)
        
        # Compute embeddings
        with Timer("Model inference time") as model_timer:
            query_embeddings = compute_embeddings(model, batch_tensor)
        
        # Find similarities
        with Timer("Similarity search time") as similarity_timer:
            top_indices, top_similarities = find_top_similarities(
                query_embeddings, precomputed_embeddings, top_k
            )
    
    # Print results and create visualizations
    results = print_results(image_paths, top_indices, top_similarities, image_ids, top_k)
    
    if dataset_name:
        print("\nLoading dataset for visualization...")
        dataset = load_dataset(dataset_name, name=name, split=split)
        
        for i, img_path in enumerate(image_paths):
            output_file = embeddings_file.replace('.npy', f'_{os.path.basename(img_path)}')
            visualize_results(
                image_tensors[i][0].cpu(), dataset, 
                image_ids[top_indices[i]], top_similarities[i], output_file, top_k
            )
    
    print_timing_summary(total_timer, loading_timer, model_timer, similarity_timer, len(image_paths))
    
    return results if len(image_paths) > 1 else results[0]

def validate_files(image_paths, embeddings_file, image_ids_file):
    """Validate that all required files exist."""
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found")
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file {embeddings_file} not found")
    
    if not os.path.exists(image_ids_file):
        raise FileNotFoundError(f"Image IDs file {image_ids_file} not found")

def main():
    """Main function to handle command line arguments and execute similarity search."""
    parser = argparse.ArgumentParser(description="Find similar images using precomputed embeddings")
    parser.add_argument("--images", type=str, nargs='+', required=True, 
                       help="Path(s) to query image(s)")
    parser.add_argument("--embeddings", type=str, required=True, 
                       help="Path to precomputed embeddings (.npy file)")
    parser.add_argument("--image_ids", type=str, required=True, 
                       help="Path to image IDs (.npy file)")
    parser.add_argument("--dataset", type=str, 
                       help="HuggingFace dataset name for visualization")
    parser.add_argument("--name", type=str, default=None, 
                       help="Dataset (subset) name")
    parser.add_argument("--split", type=str, default="test", 
                       help="Dataset split used for embeddings")
    parser.add_argument("--top_k", type=int, default=5, 
                       help="Number of similar images to retrieve")
    
    args = parser.parse_args()
    
    try:
        validate_files(args.images, args.embeddings, args.image_ids)
        find_similar_images(
            args.images, args.embeddings, args.image_ids, 
            args.dataset, args.name, args.split, args.top_k
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 