import torch
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datasets import load_dataset
import os

def find_similar_images(image_path, embeddings_file, image_ids_file, dataset_name=None, split='val', top_k=5):
    """Find the most similar images to a query image using precomputed embeddings."""
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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

    # Use this transform to test if it still finds the image
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(150),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAffine(degrees=10, translate=[0.1, 0.2], scale=[0.8, 0.9], shear=5),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load and process query image
    print(f"Processing query image: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Save transformed image for visualization
    transformed_image_for_viz = image_tensor[0].cpu()
    
    # Compute embedding for query image
    with torch.no_grad():
        query_embedding = model(image_tensor).cpu().numpy()
    
    # Compute similarities
    print("Computing similarities...")
    similarities = cosine_similarity(query_embedding, precomputed_embeddings)[0]
    
    # Find top-k most similar images
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_indices]
    top_image_ids = image_ids[top_indices]
    
    # Display results
    print(f"\nTop {top_k} most similar images:")
    for i, (idx, sim, img_id) in enumerate(zip(top_indices, top_similarities, top_image_ids)):
        print(f"{i+1}. Image ID: {img_id}, Similarity: {sim:.4f}")
    
    # Visualize results if dataset is provided
    if dataset_name:
        print("Loading dataset for visualization...")
        dataset = load_dataset(dataset_name, split=split)
        output_file = embeddings_file.replace('.npy', f'_{os.path.basename(image_path)}')
        visualize_results(transformed_image_for_viz, dataset, top_image_ids, top_similarities, output_file)
    
    return top_indices, top_similarities, top_image_ids

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
            axes[pos].set_title(f"#{i+1} Similar\nID: {img_id}\nSim: {sim:.3f}", fontsize=10)
            axes[pos].axis('off')
        except Exception as e:
            axes[pos].text(0.5, 0.5, f"Error loading\nimage {img_id}", 
                          ha='center', va='center', transform=axes[pos].transAxes)
            axes[pos].set_title(f"#{i+1} Similar\nID: {img_id}\nSim: {sim:.3f}", fontsize=10)
            axes[pos].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar images using precomputed embeddings")
    parser.add_argument("--image", type=str, required=True, help="Path to query image")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to precomputed embeddings (.npy file)")
    parser.add_argument("--image_ids", type=str, required=True, help="Path to image IDs (.npy file)")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name for visualization")
    parser.add_argument("--split", type=str, default="val", help="Dataset split used for embeddings")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to retrieve")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        exit(1)
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file {args.embeddings} not found")
        exit(1)
    if not os.path.exists(args.image_ids):
        print(f"Error: Image IDs file {args.image_ids} not found")
        exit(1)
    
    find_similar_images(args.image, args.embeddings, args.image_ids, args.dataset, args.split, args.top_k) 