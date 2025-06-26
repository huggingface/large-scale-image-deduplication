import torch
import numpy as np
import argparse
import time
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def compute_embeddings(dataset_name, name=None, split='val', output_dir='embeddings', batch_size=512):
    """Compute embeddings for all images in a HuggingFace dataset."""
    
    function_start_time = time.time()
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset and model
    dataset = load_dataset(dataset_name, name=name, split=split)
    model = torch.jit.load("models/sscd_disc_mixup.torchscript.pt")
    model.eval()
    model = model.to(device)  # Move model to GPU
    
    # Setup transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Custom collate function for DataLoader
    def collate_fn(batch):
        images = []
        indices = []
        for i, item in enumerate(batch):
            try:
                image = item['image'].convert('RGB')
                images.append(transform(image))
                indices.append(item['__index_level_0__'] if '__index_level_0__' in item else len(images)-1)
            except:
                continue
        if images:
            return torch.stack(images), indices
        return None, []
    
    # Create DataLoader for efficient batching
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images in batches
    embeddings_list = []
    image_ids = []
    
    start_time = time.time()
    model_inference_time = 0.0
    with torch.no_grad():
        for batch_tensor, batch_indices in tqdm(dataloader, desc="Computing embeddings"):
            if batch_tensor is not None:
                batch_tensor = batch_tensor.to(device)  # Move batch to GPU
                start_model_time = time.time()
                embeddings = model(batch_tensor)
                end_model_time = time.time()
                model_inference_time += end_model_time - start_model_time
                embeddings_list.append(embeddings.cpu().numpy())  # Move back to CPU before saving
                image_ids.extend(batch_indices)
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings_list)
    end_time = time.time()
    
    # Calculate timing metrics
    total_time = end_time - start_time
    time_per_sample = total_time / len(all_embeddings) if len(all_embeddings) > 0 else 0
    model_time_per_sample = model_inference_time / len(all_embeddings) if len(all_embeddings) > 0 else 0
    
    # Sanitize dataset name for filename
    sanitized_dataset_name = dataset_name.replace('/', '-')
    
    # Save embeddings and metadata
    np.save(os.path.join(output_dir, f'{sanitized_dataset_name}_{split}_embeddings.npy'), all_embeddings)
    np.save(os.path.join(output_dir, f'{sanitized_dataset_name}_{split}_image_ids.npy'), np.array(image_ids))
    
    print(f"Saved {len(all_embeddings)} embeddings to {output_dir}")
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Total time: {total_time:.5f} seconds")
    print(f"Time per sample: {time_per_sample:.5f} seconds")
    print(f"Model inference time: {model_inference_time:.5f} seconds")
    print(f"Model time per sample: {model_time_per_sample:.5f} seconds")
    
    function_end_time = time.time()
    function_total_time = function_end_time - function_start_time
    print(f"Total function time: {function_total_time:.5f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for HuggingFace dataset")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to process")
    parser.add_argument("--name", type=str, default=None, help="Dataset (subset) name")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for processing")
    
    args = parser.parse_args()
    compute_embeddings(args.dataset, args.name, args.split, args.output_dir, args.batch_size) 