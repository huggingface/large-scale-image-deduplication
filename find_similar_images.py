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

def find_similar_images(image_paths, embeddings_file, image_ids_file, dataset_name=None, split='val', top_k=5):
    """Find the most similar images to one or two query images using precomputed embeddings."""
    
    # Start total timing
    total_start_time = time.time()
    
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Start loading timing
    loading_start_time = time.time()
    
    # Load precomputed embeddings and image IDs
    print("Loading precomputed embeddings...")
    precomputed_embeddings = np.load(embeddings_file)
    image_ids = np.load(image_ids_file)
    print(f"Loaded {len(precomputed_embeddings)} precomputed embeddings")
    
    # Load model
    print("Loading model...")
    model = torch.jit.load("models/sscd_disc_mixup.torchscript.pt")
    model.eval()
    model = model.to(device)
    
    # Setup transforms (same as in compute_embeddings.py)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    # Same transform as for the embeddings
    # transform = transforms.Compose([
    #     transforms.Resize([320, 320]),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # Use this transform to test if it still finds the image (includes random transforms, so even when given the same image twice it will produce different results)
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(150),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAffine(degrees=10, translate=[0.1, 0.2], scale=[0.8, 0.9], shear=5),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load and process query images
    print(f"Processing {len(image_paths)} query image(s)")
    image_tensors = []
    for img_path in image_paths:
        original_image = Image.open(img_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0)
        image_tensors.append(image_tensor)
    
    # Stack all images into a batch
    batch_tensor = torch.cat(image_tensors, dim=0).to(device)

    loading_time = time.time() - loading_start_time
    print(f"Loading time: {loading_time:.5f} seconds")
    
    # Compute embeddings for all query images
    print(f"\nComputing embeddings for {len(image_paths)} image(s)...")
    model_start_time = time.time()
    with torch.no_grad():
        query_embeddings = model(batch_tensor).cpu().numpy()
    model_time = time.time() - model_start_time
    print(f"Model inference time: {model_time:.5f} seconds ({model_time/len(image_paths):.5f} seconds per image)")
    
    # Compute similarities for all query images at once (batched)
    print(f"Computing similarities for all {len(image_paths)} image(s)...")
    similarity_start_time = time.time()
    all_similarities = cosine_similarity(query_embeddings, precomputed_embeddings)
    
    # Batch process top-k selection for all images
    all_top_indices = np.argsort(all_similarities, axis=1)[:, ::-1][:, :top_k]  # Shape: (num_images, top_k)
    all_top_similarities = np.take_along_axis(all_similarities, all_top_indices, axis=1)  # Shape: (num_images, top_k)
    all_top_image_ids = image_ids[all_top_indices]  # Shape: (num_images, top_k)
    
    similarity_time = time.time() - similarity_start_time
    print(f"Similarity search time: {similarity_time:.5f} seconds ({similarity_time/len(image_paths):.5f} seconds per image)")
    
    # Only loop for printing and visualization (non-core functionality)
    results = []
    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing query image {i+1}: {img_path}")
        
        top_indices = all_top_indices[i]
        top_similarities = all_top_similarities[i] 
        top_image_ids = all_top_image_ids[i]
        
        print(f"Top {top_k} most similar images:")
        for j, (idx, sim, img_id) in enumerate(zip(top_indices, top_similarities, top_image_ids)):
            print(f"  {j+1}. Image ID: {img_id}, Similarity: {sim:.4f}")
        
        results.append((top_indices, top_similarities, top_image_ids))
        
        # Visualize results if dataset is provided
        if dataset_name:
            if i == 0:  # Load dataset only once
                print("Loading dataset for visualization...")
                dataset = load_dataset(dataset_name, split=split)
            output_file = embeddings_file.replace('.npy', f'_{os.path.basename(img_path)}')
            visualize_results(image_tensors[i][0].cpu(), dataset, top_image_ids, top_similarities, output_file)
    
    # Print timing summary
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.5f} seconds")
    print(f"Loading time: {loading_time:.5f} seconds ({loading_time/total_time*100:.1f}%)")
    print(f"Model inference time: {model_time:.5f} seconds ({model_time/total_time*100:.1f}%)")
    print(f"Similarity search time: {similarity_time:.5f} seconds ({similarity_time/total_time*100:.1f}%)")
    print(f"Average similarity search per image: {similarity_time/len(image_paths):.5f} seconds")
    
    return results if len(image_paths) > 1 else results[0]

def denormalize_image(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_results(query_tensor, dataset, top_image_ids, top_similarities, output_file):
    """Visualize the query image and top similar images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Denormalize and convert query image for display
    query_img = denormalize_image(query_tensor)
    query_img = torch.clamp(query_img, 0, 1)
    query_img = query_img.permute(1, 2, 0).numpy()
    
    # Display query image
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image\n(After Transform)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Display top 5 similar images
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, (pos, img_id, sim) in enumerate(zip(positions, top_image_ids, top_similarities)):
        try:
            # Get image from dataset
            similar_img = dataset[int(img_id)]['image'].convert('RGB')
            
            # Display image
            axes[pos].imshow(similar_img)
            axes[pos].set_title(f"#{i+1} Similar\nID: {img_id}\nSim: {sim:.5f}", fontsize=10)
            axes[pos].axis('off')
        except Exception as e:
            axes[pos].text(0.5, 0.5, f"Error loading\nimage {img_id}", 
                          ha='center', va='center', transform=axes[pos].transAxes)
            axes[pos].set_title(f"#{i+1} Similar\nID: {img_id}\nSim: {sim:.5f}", fontsize=10)
            axes[pos].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar images using precomputed embeddings")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Path(s) to query image(s)")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to precomputed embeddings (.npy file)")
    parser.add_argument("--image_ids", type=str, required=True, help="Path to image IDs (.npy file)")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name for visualization")
    parser.add_argument("--split", type=str, default="val", help="Dataset split used for embeddings")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to retrieve")
    
    args = parser.parse_args()
    
    # Check if files exist
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image file {img_path} not found")
            exit(1)
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file {args.embeddings} not found")
        exit(1)
    if not os.path.exists(args.image_ids):
        print(f"Error: Image IDs file {args.image_ids} not found")
        exit(1)
    
    find_similar_images(args.images, args.embeddings, args.image_ids, args.dataset, args.split, args.top_k) 