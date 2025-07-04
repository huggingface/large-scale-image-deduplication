import torch
import numpy as np
import argparse
import time
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageCollator:
    """Collator class for processing image batches with transforms."""
    
    def __init__(self, transform, deduplicate=False, id_column='image_id'):
        self.transform = transform
        self.deduplicate = deduplicate
        self.id_column = id_column
        self.seen_ids = set() if deduplicate else None
    
    def _is_valid_image(self, img):
        """Check if the object is a valid image that can be processed."""
        if img is None:
            return False
        
        try:
            # Try to actually load/verify the image to catch truncated images early
            img.load()
            # Test if we can convert it to RGB
            _ = img.convert('RGB')
            return True
        except (OSError, IOError, Image.DecompressionBombError, Exception) as e:
            print(f"Warning: Invalid or corrupted image detected: {e}")
            return False
    
    def __call__(self, batch):
        images = []
        indices = []
        for item in batch:
            try:
                # Get the explicit dataset index we added
                dataset_idx = item.get('dataset_idx', None)
                if dataset_idx is None:
                    print(f"Warning: No dataset_idx found in item")
                    continue
                
                # Check for deduplication
                if self.deduplicate and self.id_column in item:
                    image_id = item[self.id_column]
                    if image_id in self.seen_ids:
                        continue
                    self.seen_ids.add(image_id)
                
                # Handle different image key patterns
                image_data = None
                collected_images = []
                
                if 'image' in item:
                    image_data = item['image']
                    if isinstance(image_data, list):
                        # Handle list of images
                        for img in image_data:
                            if self._is_valid_image(img):
                                collected_images.append(img)
                    else:
                        # Handle single image
                        if self._is_valid_image(image_data):
                            collected_images.append(image_data)
                elif 'images' in item:
                    image_data = item['images']
                    if isinstance(image_data, list):
                        for img in image_data:
                            if self._is_valid_image(img):
                                collected_images.append(img)
                    else:
                        if self._is_valid_image(image_data):
                            collected_images.append(image_data)
                elif any(key.startswith('image_') and key[6:].isdigit() for key in item.keys()):
                    # Handle numbered image keys like 'image_0', 'image_1', etc.
                    numbered_image_keys = [key for key in item.keys() if key.startswith('image_') and key[6:].isdigit()]
                    # Sort keys by number to maintain order
                    numbered_image_keys.sort(key=lambda x: int(x.split('_')[1]))
                    for key in numbered_image_keys:
                        img_data = item[key]
                        if self._is_valid_image(img_data):
                            collected_images.append(img_data)
                else:
                    print(f"Warning: No image keys found in item. Available keys: {list(item.keys())}")
                    continue
                
                # Process all collected images
                for img in collected_images:
                    if self._is_valid_image(img):
                        try:
                            processed_img = img.convert('RGB')
                            images.append(self.transform(processed_img))
                            indices.append(dataset_idx)
                        except Exception as img_error:
                            print(f"Error processing individual image: {img_error}")
                
            except Exception as e:
                print(f"Error processing item: {e}")
                
        if images:
            return torch.stack(images), indices
        return None, []


def setup_device():
    """Setup and return the appropriate device (GPU/CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_model(model_path="models/sscd_disc_mixup.torchscript.pt", device=None):
    """Load and setup the model."""
    model = torch.jit.load(model_path)
    model.eval()
    if device:
        model = model.to(device)
    return model


def create_transforms():
    """Create and return image transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])


def compute_batch_embeddings(model, dataloader, device):
    """Compute embeddings for all batches and return results with timing info."""
    embeddings_list = []
    image_ids = []
    model_inference_time = 0.0
    
    with torch.no_grad():
        for batch_tensor, batch_indices in tqdm(dataloader, desc="Computing embeddings"):
            if batch_tensor is not None:
                batch_tensor = batch_tensor.to(device)
                start_model_time = time.time()
                embeddings = model(batch_tensor)
                end_model_time = time.time()
                model_inference_time += end_model_time - start_model_time
                embeddings_list.append(embeddings.cpu().numpy())
                image_ids.extend(batch_indices)
    
    return embeddings_list, image_ids, model_inference_time


def save_results(embeddings, image_ids, dataset_name, split, output_dir, name=None):
    """Save embeddings and image IDs to files."""
    os.makedirs(output_dir, exist_ok=True)
    sanitized_dataset_name = dataset_name.replace('/', '-')
    
    # Include name in filename if provided
    if name:
        sanitized_name = name.replace('/', '-')
        filename_base = f'{sanitized_dataset_name}_{sanitized_name}_{split}'
    else:
        filename_base = f'{sanitized_dataset_name}_{split}'
    
    np.save(os.path.join(output_dir, f'{filename_base}_embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, f'{filename_base}_image_ids.npy'), np.array(image_ids))


def print_results(embeddings, total_time, model_inference_time, output_dir):
    """Print timing and result statistics."""
    num_embeddings = len(embeddings)
    time_per_sample = total_time / num_embeddings if num_embeddings > 0 else 0
    model_time_per_sample = model_inference_time / num_embeddings if num_embeddings > 0 else 0
    
    print(f"Saved {num_embeddings} embeddings to {output_dir}/")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Total time: {total_time:.5f} seconds")
    print(f"Time per sample: {time_per_sample:.5f} seconds")
    print(f"Model inference time: {model_inference_time:.5f} seconds")
    print(f"Model time per sample: {model_time_per_sample:.5f} seconds")


def compute_embeddings(dataset_name, name=None, split='test', output_dir='embeddings-lmms', batch_size=32, deduplicate=False, id_column='image_id'):
    """Compute embeddings for all images in a HuggingFace dataset."""
    function_start_time = time.time()
    
    # Setup components
    device = setup_device()
    model = load_model(device=device)
    transform = create_transforms()
    collator = ImageCollator(transform, deduplicate=deduplicate, id_column=id_column)
    
    # Load dataset and add explicit indices
    dataset = load_dataset(dataset_name, name=name, split=split)
    dataset = dataset.add_column("dataset_idx", list(range(len(dataset))))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=8)
    
    # Compute embeddings
    start_time = time.time()
    embeddings_list, image_ids, model_inference_time = compute_batch_embeddings(model, dataloader, device)
    end_time = time.time()
    
    # Process results
    all_embeddings = np.vstack(embeddings_list)
    total_time = end_time - start_time
    
    # Save and report results
    save_results(all_embeddings, image_ids, dataset_name, split, output_dir, name)
    print_results(all_embeddings, total_time, model_inference_time, output_dir)
    
    function_end_time = time.time()
    function_total_time = function_end_time - function_start_time
    print(f"Total function time: {function_total_time:.5f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for HuggingFace dataset")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to process")
    parser.add_argument("--name", type=str, default=None, help="Dataset (subset) name")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--deduplicate", action="store_true", help="Enable deduplication based on ID column")
    parser.add_argument("--id_column", type=str, default="image_id", help="Column name for image ID (default: image_id)")
    
    args = parser.parse_args()
    compute_embeddings(args.dataset, args.name, args.split, args.output_dir, args.batch_size, args.deduplicate, args.id_column) 