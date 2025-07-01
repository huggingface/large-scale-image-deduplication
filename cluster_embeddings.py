import numpy as np
import argparse
import pickle
import random
from collections import defaultdict

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from umap import UMAP
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset


def cluster_embeddings(embeddings_file, n_components=2, metric="cosine", 
                      eps=0.5, min_samples=5, n_jobs=16, output_file=None):
    """Cluster embeddings using UMAP for dimensionality reduction and DBSCAN for clustering."""
    
    # Load and reduce embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    umap_reducer = UMAP(n_components=n_components, metric=metric)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    print(f"Reduced to {n_components}D using UMAP")
    
    # Cluster with DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    labels = clusterer.fit_predict(reduced_embeddings)
    
    # Analyze results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
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


def plot_clusters(cluster_data, cluster_labels=None, dataset=None, output_file=None, figsize=(10, 8)):
    """Plot clustering results with optional semantic labels."""
    
    labels = cluster_data['labels']
    embeddings = cluster_data['reduced_embeddings']
    n_clusters = cluster_data['n_clusters']
    n_noise = cluster_data['n_noise']
    
    plt.figure(figsize=figsize, dpi=300)
    
    # Plot noise points
    noise_mask = labels == -1
    if noise_mask.any():
        plt.scatter(embeddings[noise_mask, 0], embeddings[noise_mask, 1], 
                   c="lightgray", s=0.5, alpha=0.3, label="Noise")
    
    # Plot clusters
    clustered_mask = labels != -1
    if clustered_mask.any():
        plt.scatter(embeddings[clustered_mask, 0], embeddings[clustered_mask, 1], 
                   c=labels[clustered_mask], s=1, alpha=0.8, cmap="tab10")
    
    # Add semantic labels if provided
    if cluster_labels:
        _add_cluster_labels(embeddings, labels, cluster_labels)
    
    if dataset is None:
        plt.title(f"{n_clusters} clusters, {len(labels)-n_noise}/{len(labels)} points clustered")
    else:
        plt.title(f"{dataset}\n{n_clusters} clusters, {len(labels)-n_noise}/{len(labels)} points clustered")
    plt.axis('off')
    
    if output_file is None:
        if dataset is not None:
            sanitized_dataset = dataset.replace('/', '-')
            output_file = f'clustering_{sanitized_dataset}.png'
        else:
            output_file = 'clustering.png'
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()


def _add_cluster_labels(embeddings, labels, cluster_labels):
    """Add text labels to cluster centers."""
    # Calculate cluster centers
    centers = {}
    for label in set(labels):
        if label != -1 and label in cluster_labels:
            mask = labels == label
            centers[label] = (np.mean(embeddings[mask, 0]), np.mean(embeddings[mask, 1]))
    
    # Add text annotations
    for label, (x, y) in centers.items():
        text = plt.text(x, y, cluster_labels[label], ha='center', va='center', 
                       fontsize=4, alpha=0.8, weight='normal')
        text.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0, boxstyle='round,pad=0.2'))


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


def _sample_cluster_data(cluster_data, dataset, image_ids, label, n_examples):
    """Sample images and questions from a specific cluster."""
    labels = cluster_data['labels']
    cluster_indices = [i for i, l in enumerate(labels) if l == label]
    
    n_samples = min(n_examples, len(cluster_indices))
    sampled_indices = random.sample(cluster_indices, n_samples)
    
    images, questions = [], []
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


def _generate_label_with_vlm(images, questions, model, processor, device):
    """Generate semantic label using vision-language model."""
    if not images:
        return None
    
    questions_context = "\n".join([f"Image {i+1}: {q}" for i, q in enumerate(questions)])
    
    prompt = (
        f"Do not answer the questions in the texts you are given."
        f"Focus on the underlying concepts of the visual content and question. Answer in the format: Word1, Word2"
        f"Analyze these {len(images)} clustered images and their contexts:\n{questions_context}\n\n"
        f"Provide exactly 2 comma-separated words describing the main themes that distinguish "
        f"this cluster. Answer in the format: Word1, Word2"
    )
    
    # Prepare conversation with all images
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, 
                                         tokenize=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=50, temperature=0.1, 
                              do_sample=True, pad_token_id=processor.tokenizer.eos_token_id)
    
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant\n" in response:
        label = response.split("assistant\n")[-1].strip()
    else:
        label = response.strip()
    
    return label.split("\n")[0].split(".")[0].strip()


def generate_cluster_labels(cluster_data, dataset_name, subset_name=None, split='test', 
                          n_examples=10, vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Generate semantic labels for clusters using a vision-language model."""
    
    # Load dataset and image IDs
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, name=subset_name, split=split)
    image_ids = _load_image_ids(cluster_data['embeddings_file'])
    
    # Load VL model
    print(f"Loading model: {vl_model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(vl_model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        vl_model_name, torch_dtype=torch.float16).to(device)
    
    # Generate labels for each cluster
    labels = cluster_data['labels']
    unique_labels = [l for l in set(labels) if l != -1]
    cluster_labels = {-1: "Noise"}
    
    print(f"Generating labels for {len(unique_labels)} clusters...")
    
    for label in unique_labels:
        print(f"Processing cluster {label}...")
        
        images, questions = _sample_cluster_data(cluster_data, dataset, image_ids, label, n_examples)
        
        if images:
            semantic_label = _generate_label_with_vlm(images, questions, model, processor, device)
            cluster_labels[label] = semantic_label or f"Cluster_{label}"
        else:
            cluster_labels[label] = f"Cluster_{label}"
    
    return cluster_labels


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings using UMAP + DBSCAN")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file")
    
    # Clustering parameters
    parser.add_argument("--n-components", type=int, default=2, help="UMAP dimensions")
    parser.add_argument("--metric", default="cosine", help="UMAP distance metric")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--n-jobs", type=int, default=16, help="Number of parallel jobs")
    parser.add_argument("--output", help="Output file path")
    
    # Semantic labeling
    parser.add_argument("--generate-labels", action="store_true", help="Generate semantic labels")
    parser.add_argument("--dataset", help="Dataset name (required for labeling)")
    parser.add_argument("--subset", help="Dataset subset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--n-examples", type=int, default=5, help="Examples per cluster for labeling")
    parser.add_argument("--vl-model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Vision-language model")
    
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
                args.n_examples, args.vl_model
            )
        except Exception as e:
            print(f"Warning: Failed to generate labels: {e}")
    
    # Generate plot
    plot_clusters(cluster_data, semantic_labels, args.dataset, args.output)


if __name__ == "__main__":
    main()