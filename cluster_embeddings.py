import numpy as np
import argparse
import pickle
import random
from collections import defaultdict
import gc
import torch

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from umap import UMAP
import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset


def cluster_embeddings(embeddings_file, n_components=2, metric="cosine", 
                      eps=0.5, min_samples=5, n_jobs=12, output_file=None):
    """Cluster embeddings using UMAP for dimensionality reduction and DBSCAN for clustering."""
    
    # Load and reduce embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Optimize UMAP parameters for performance
    umap_reducer = UMAP(
        n_components=n_components, 
        metric=metric,
        low_memory=True,  # Use less memory
        n_jobs=n_jobs     # Parallel processing
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    print(f"Reduced to {n_components}D using UMAP")
    
    # Cluster with DBSCAN - already optimized with n_jobs
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    labels = clusterer.fit_predict(reduced_embeddings)
    
    # Analyze results - vectorized operations
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)  # Vectorized count
    
    print(f"Found {n_clusters} clusters with {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    
    # Calculate silhouette score for valid clusters
    silhouette = None
    if n_clusters > 1:
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            silhouette = silhouette_score(reduced_embeddings[non_noise_mask], labels[non_noise_mask])
            print(f"Silhouette score: {silhouette:.3f}")
    
    # Save results
    if output_file is None:
        output_file = embeddings_file.replace('_embeddings.npy', '_clusters.pkl')
    
    cluster_data = {
        'labels': labels,
        'reduced_embeddings': reduced_embeddings,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette,
        'embeddings_file': embeddings_file,
        'umap_reducer': umap_reducer
    }
    
    # with open(output_file, 'wb') as f:
    #     pickle.dump(cluster_data, f)
    
    # print(f"Results saved to {output_file}")
    return cluster_data


def plot_clusters(cluster_data, cluster_labels=None, dataset=None, output_file=None, figsize=(12, 9), max_clusters=10):
    """Plot clustering results with optional semantic labels."""
    
    labels = cluster_data['labels']
    embeddings = cluster_data['reduced_embeddings']
    n_clusters = cluster_data['n_clusters']
    n_noise = cluster_data['n_noise']
    
    # Set style for prettier plots
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Define a prettier color palette
    colors = plt.cm.Set3(np.linspace(0, 1, max(12, n_clusters)))
    
    # Plot noise points with subtle styling - vectorized
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(embeddings[noise_mask, 0], embeddings[noise_mask, 1], 
                  c="#404040", s=10, alpha=0.5, label="Noise", edgecolors='none')
    
    # Plot clusters with prettier colors and styling - optimized loop
    clustered_mask = labels != -1
    if np.any(clustered_mask):
        unique_labels = np.unique(labels[clustered_mask])
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            ax.scatter(embeddings[cluster_mask, 0], embeddings[cluster_mask, 1], 
                      c=[colors[i % len(colors)]], s=25, alpha=0.8, 
                      edgecolors='white', linewidth=0.1)
    
    # Add semantic labels if provided
    if cluster_labels:
        _add_cluster_labels(embeddings, labels, cluster_labels, ax, max_clusters)
    
    # Style the title and layout
    #title_text = f"{n_clusters} clusters • {len(labels)-n_noise:,}/{len(labels):,} points clustered"
    if dataset:
        title_text = f"{dataset}" #\n{title_text}"
    
    ax.set_title(title_text, fontsize=14, color='white', fontweight='300', pad=20)
    ax.axis('off')
    
    # Remove any remaining axes elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if output_file is None:
        if dataset is not None:
            sanitized_dataset = dataset.replace('/', '-')
            output_file = f'clustering_{sanitized_dataset}.png'
        else:
            output_file = 'clustering.png'
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='#1a1a1a')
    print(f"Plot saved to {output_file}")
    plt.close()


def _validate_semantic_label(label):
    """Validate if a semantic label is properly formatted and not hallucinated."""
    if not label or not isinstance(label, str):
        return False
    
    label = label.strip()
    
    # Filter out default cluster labels
    if label.startswith("Cluster_") or label == "Noise":
        return False
    
    # Check for comma-separated format (Word1, Word2)
    parts = [part.strip() for part in label.split(',')]
    
    # Should have 1-3 parts, each being reasonable words
    if len(parts) < 1 or len(parts) > 3:
        return False
    
    # Each part should be 1-3 words, reasonable length
    for part in parts:
        words = part.split()
        if len(words) < 1 or len(words) > 3:
            return False
        if len(part) > 25:  # Too long, likely hallucinated
            return False
        if not all(word.replace('-', '').replace('_', '').isalpha() for word in words):
            return False  # Contains non-alphabetic characters
    
    return True


def _add_cluster_labels(embeddings, labels, cluster_labels, ax, max_clusters=10):
    """Add text labels to cluster centers for the largest clusters."""
    # Calculate cluster sizes - vectorized
    unique_labels = np.unique(labels)
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[label] = np.sum(labels == label)
    
    # Get the top max_clusters largest clusters
    largest_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:max_clusters]
    largest_cluster_labels = [label for label, size in largest_clusters]
    
    # Calculate cluster centers and filter valid labels for only the largest clusters - vectorized
    centers = {}
    for label in largest_cluster_labels:
        if label in cluster_labels:
            semantic_label = cluster_labels[label]
            # Only add labels that pass validation
            if _validate_semantic_label(semantic_label):
                mask = labels == label
                centers[label] = (np.mean(embeddings[mask, 0]), np.mean(embeddings[mask, 1]), semantic_label)
    
    # Add text annotations
    for label, (x, y, text_label) in centers.items():
        text = ax.text(x, y, text_label, ha='center', va='center', 
                       fontsize=7, alpha=0.8, color='black')
        text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0, boxstyle='round,pad=0.2'))


def _load_image_ids(embeddings_file):
    """Load corresponding image IDs for the embeddings."""
    image_ids_file = embeddings_file.replace('_embeddings.npy', '_image_ids.npy')
    try:
        return np.load(image_ids_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image IDs file not found: {image_ids_file}")


def _extract_image_from_item(item):
    """Extract image data from a dataset item, handling various formats."""
    # Try different possible image keys
    for key in ['image', 'images']:
        if key in item:
            image_data = item[key]
            if isinstance(image_data, list):
                image_data = image_data[0]
            return image_data.convert('RGB') if hasattr(image_data, 'convert') else image_data
    
    # Try numbered image keys (image_0, image_1, etc.)
    numbered_keys = sorted([k for k in item.keys() if k.startswith('image_') and k[6:].isdigit()],
                          key=lambda x: int(x.split('_')[1]))
    if numbered_keys:
        return item[numbered_keys[0]].convert('RGB')
    
    return None


def _extract_question_from_item(item):
    """Extract question/text from a dataset item."""
    for key in ['question', 'text', 'query', 'prompt']:
        if key in item:
            return item[key]
    return "No question available"


def _sample_cluster_data_batch(cluster_data, dataset, image_ids, cluster_labels, n_examples):
    """Sample images and questions from multiple clusters efficiently."""
    labels = cluster_data['labels']
    all_cluster_data = {}
    
    # Pre-compute all cluster indices
    cluster_indices = {}
    for label in cluster_labels:
        if label != -1:
            cluster_indices[label] = np.where(labels == label)[0]  # Vectorized
    
    # Sample indices for all clusters
    sampled_indices_all = set()
    cluster_samples = {}
    
    for label in cluster_labels:
        if label == -1:
            continue
            
        indices = cluster_indices[label]
        n_samples = min(n_examples, len(indices))
        sampled_indices = np.random.choice(indices, n_samples, replace=False)
        cluster_samples[label] = sampled_indices
        sampled_indices_all.update(sampled_indices)
    
    # Convert to list and get original indices
    sampled_indices_list = list(sampled_indices_all)
    original_indices = [int(image_ids[idx]) for idx in sampled_indices_list]
    
    # Batch load dataset items
    dataset_items = dataset.select(original_indices)
    
    # Process items by cluster
    for label, sampled_indices in cluster_samples.items():
        images, questions = [], []
        
        for idx in sampled_indices:
            try:
                original_idx = int(image_ids[idx])
                # Find the item in our batch
                item_idx = original_indices.index(original_idx)
                item = dataset_items[item_idx]
                
                image = _extract_image_from_item(item)
                question = _extract_question_from_item(item)
                
                if image is not None:
                    images.append(image)
                    questions.append(question)
                    
            except Exception as e:
                print(f"Warning: Failed to process image {idx}: {e}")
                continue
        
        all_cluster_data[label] = (images, questions)
    
    return all_cluster_data


def _sample_cluster_data(cluster_data, dataset, image_ids, label, n_examples):
    """Sample images and questions from a specific cluster."""
    labels = cluster_data['labels']
    cluster_indices = np.where(labels == label)[0]  # Vectorized
    
    n_samples = min(n_examples, len(cluster_indices))
    sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)
    
    images, questions = [], []
    original_indices = [int(image_ids[idx]) for idx in sampled_indices]
    
    # Batch load items for efficiency
    try:
        dataset_items = dataset.select(original_indices)
        for i, item in enumerate(dataset_items):
            image = _extract_image_from_item(item)
            question = _extract_question_from_item(item)
            
            if image is not None:
                images.append(image)
                questions.append(question)
    except Exception as e:
        print(f"Warning: Failed to batch process cluster {label}: {e}")
        # Fallback to individual processing
        for idx in sampled_indices:
            try:
                original_idx = int(image_ids[idx])
                item = dataset[original_idx]
                
                image = _extract_image_from_item(item)
                question = _extract_question_from_item(item)
                
                if image is not None:
                    images.append(image)
                    questions.append(question)
                    
            except Exception as e:
                print(f"Warning: Failed to process image {idx}: {e}")
                continue
    
    return images, questions


def _get_dataset_specific_guidance(dataset_name):
    """Get dataset-specific guidance for better label generation."""
    dataset_guidance = {
        'MMMU': 'Focus on academic subjects (math, physics, chemistry, biology, history, etc.) and question types',
        'ScienceQA': 'Focus on science domains (physics, chemistry, biology, earth science) and question formats',
        'ChartQA': 'Focus on chart types (bar, line, pie, scatter) and data domains (economics, demographics, etc.)',
        'VQAv2': 'Focus on visual scenes, objects, activities, and spatial relationships',
        'TextVQA': 'Focus on text content types (signs, documents, books, etc.) and visual contexts',
        'COCO': 'Focus on real-world scenes, common objects, activities, and settings',
        'A-OKVQA': 'Focus on knowledge domains and visual reasoning types',
        'GQA': 'Focus on visual relationships, object attributes, and spatial reasoning',
    }
    
    # Extract base dataset name (remove org prefix if present)
    base_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    base_name = base_name.replace('lmms-lab-', '').replace('_', '')
    
    for key, guidance in dataset_guidance.items():
        if key.lower() in base_name.lower():
            return f"Dataset context: This is {key} data. {guidance}.\n"
    
    return "Dataset context: General multimodal dataset.\n"


def _generate_label_with_vlm(images, questions, model, processor, device, dataset_name=None):
    """Generate semantic label using vision-language model."""
    if not images:
        return None
    
    questions_context = "\n".join([f"Image {i+1}: {q}" for i, q in enumerate(questions)])
    dataset_guidance = _get_dataset_specific_guidance(dataset_name) if dataset_name else ""
    
    prompt = (
        f"You are analyzing {len(images)} images that have been grouped together by a clustering algorithm. "
        f"Your task is to identify the main visual themes that make these images similar to each other.\n\n"
        f"{dataset_guidance}"
        f"Image contexts:\n{questions_context}\n\n"
        f"Based on the visual content and associated contexts, identify exactly 2 specific themes that "
        f"best describe what makes this cluster distinct. Focus on:\n"
        f"- Visual subjects (objects, people, animals, scenes)\n"
        f"- Content domains (science, geography, art, food, etc.)\n"
        f"- Visual styles or formats (charts, diagrams, photos, etc.)\n\n"
        f"Examples of good labels:\n"
        f"- 'Charts, Data' (for visualization/statistical content)\n"
        f"- 'Food, Cooking' (for culinary images)\n"
        f"- 'Science, Biology' (for scientific diagrams)\n"
        f"- 'Animals, Nature' (for wildlife photos)\n"
        f"- 'Architecture, Buildings' (for structural images)\n\n"
        f"Avoid generic terms like 'Image', 'Photo', 'Question', 'Text', 'Content'.\n"
        f"Respond with exactly 2 words separated by a comma: Word1, Word2"
    )
    
    # Prepare conversation with all images
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    
    conversation = [{"role": "user", "content": content}]
    
    try:
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, 
                                             tokenize=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                inputs, 
                max_new_tokens=20,  # Reduced since we only need 2 words
                temperature=0.3,    # Slightly higher for more creative labels
                do_sample=True, 
                top_p=0.8,         # Nucleus sampling for better quality
                repetition_penalty=1.1,  # Avoid repetitive outputs
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant\n" in response:
            label = response.split("assistant\n")[-1].strip()
        else:
            label = response.strip()
        
        # Clean and extract the final label
        label = label.split("\n")[0].split(".")[0].strip()
        
        # Remove any quotes or extra formatting
        label = label.strip('"\'`')
        
        # Ensure we have exactly 2 comma-separated words
        if ',' in label:
            parts = [part.strip() for part in label.split(',')]
            if len(parts) >= 2:
                return f"{parts[0]}, {parts[1]}"
        
        return label
    
    except Exception as e:
        print(f"Warning: VLM inference failed: {e}")
        return None
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_cluster_labels(cluster_data, dataset_name, subset_name=None, split='test', 
                          n_examples=10, vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct", max_clusters=10):
    """Generate semantic labels for clusters using a vision-language model."""
    
    # Load image IDs first (lightweight)
    image_ids = _load_image_ids(cluster_data['embeddings_file'])
    
    # Calculate cluster sizes and identify the largest clusters
    labels = cluster_data['labels']
    unique_labels = np.unique(labels)
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[label] = np.sum(labels == label)  # Vectorized
    
    # Get the top max_clusters largest clusters
    largest_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:max_clusters]
    largest_cluster_labels = [label for label, size in largest_clusters]
    
    print(f"Processing only the {len(largest_cluster_labels)} largest clusters (out of {len(cluster_sizes)} total)")
    
    # Initialize cluster labels with default names for all clusters
    all_unique_labels = [l for l in unique_labels if l != -1]
    cluster_labels = {-1: "Noise"}
    for label in all_unique_labels:
        cluster_labels[label] = f"Cluster_{label}"
    
    # Early return if no clusters to process
    if not largest_cluster_labels:
        return cluster_labels
    
    # Load dataset and VL model only when needed
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, name=subset_name, split=split)
    
    print(f"Loading model: {vl_model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(vl_model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        vl_model_name, torch_dtype=torch.float16).to(device)
    
    try:
        # Generate semantic labels only for the largest clusters
        print(f"Generating semantic labels for {len(largest_cluster_labels)} largest clusters...")
        
        for label in largest_cluster_labels:
            print(f"Processing cluster {label} (size: {cluster_sizes[label]})...")
            
            images, questions = _sample_cluster_data(cluster_data, dataset, image_ids, label, n_examples)
            
            if images:
                semantic_label = _generate_label_with_vlm(images, questions, model, processor, device, dataset_name)
                cluster_labels[label] = semantic_label or f"Cluster_{label}"
            else:
                cluster_labels[label] = f"Cluster_{label}"
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            del model
            del processor
            torch.cuda.empty_cache()
            gc.collect()
    
    return cluster_labels


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings using UMAP + DBSCAN")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file")
    
    # Clustering parameters
    parser.add_argument("--n-components", type=int, default=2, help="UMAP dimensions")
    parser.add_argument("--metric", default="cosine", help="UMAP distance metric")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples")
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of parallel jobs")
    parser.add_argument("--output", help="Output file path")
    
    # Semantic labeling
    parser.add_argument("--generate-labels", action="store_true", help="Generate semantic labels")
    parser.add_argument("--dataset", help="Dataset name (required for labeling)")
    parser.add_argument("--subset", help="Dataset subset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--n-examples", type=int, default=10, help="Examples per cluster for labeling")
    parser.add_argument("--vl-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Vision-language model")
    parser.add_argument("--max-clusters", type=int, default=12, help="Maximum number of largest clusters to label")
    
    args = parser.parse_args()
    
    # Perform clustering
    cluster_data = cluster_embeddings(
        args.embeddings, args.n_components, args.metric,
        args.eps, args.min_samples, args.n_jobs, args.output
    )
    
    # Generate semantic labels if requested
    semantic_labels = None
    if args.generate_labels:
        if not args.dataset:
            print("Error: --dataset required for label generation")
            return
        
        try:
            semantic_labels = generate_cluster_labels(
                cluster_data, args.dataset, args.subset, args.split,
                args.n_examples, args.vl_model, args.max_clusters
            )
        except Exception as e:
            print(f"Warning: Failed to generate labels: {e}")
    
    # Generate plot
    plot_clusters(cluster_data, semantic_labels, args.dataset, args.output, max_clusters=args.max_clusters)


if __name__ == "__main__":
    main()