import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt

def find_optimal_clusters_silhouette(embeddings, max_clusters=1000, min_clusters=1000, step=100):
    """Find optimal number of clusters using silhouette analysis."""
    silhouette_scores = []
    cluster_range = list(range(min_clusters, max_clusters + 1, step))
    
    print(f"Testing cluster numbers: {cluster_range}")
    
    for k in cluster_range:
        print(f"Testing {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"  Silhouette score for {k} clusters: {score:.3f}")
    
    # Find the k with maximum silhouette score
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = cluster_range[optimal_idx]
    
    return optimal_k, silhouette_scores, cluster_range

def plot_tuning_results(scores, cluster_range, optimal_k, output_prefix):
    """Plot the silhouette analysis results."""
    plt.figure(figsize=(12, 6))
    plt.plot(cluster_range, scores, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                label=f'Optimal k={optimal_k}')
    
    plt.title('Silhouette Analysis for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"{output_prefix}_silhouette_tuning.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Tuning plot saved to {output_file}")

def cluster_embeddings(embeddings_file, n_clusters=None, auto_tune=False, max_clusters=100, min_clusters=2, step=5, output_file=None):
    # Load embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Determine optimal number of clusters if auto-tuning is enabled
    if auto_tune:
        print(f"Auto-tuning number of clusters using silhouette analysis...")
        print(f"Testing range: {min_clusters} to {max_clusters} with step {step}")
        output_prefix = embeddings_file.replace('_embeddings.npy', '')
        
        optimal_k, scores, cluster_range = find_optimal_clusters_silhouette(
            embeddings, max_clusters, min_clusters, step)
        plot_tuning_results(scores, cluster_range, optimal_k, output_prefix)
        
        n_clusters = optimal_k
        print(f"Optimal number of clusters found: {n_clusters}")
    
    elif n_clusters is None:
        n_clusters = 10  # Default value
        print(f"Using default number of clusters: {n_clusters}")
    
    # Cluster using K-means
    print(f"Clustering with {n_clusters} clusters...")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(embeddings)
    
    # Calculate silhouette score for the final clustering
    if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
        final_silhouette = silhouette_score(embeddings, labels)
        print(f"Final silhouette score: {final_silhouette:.3f}")
    else:
        final_silhouette = None
    
    # Save results
    if output_file is None:
        output_file = embeddings_file.replace('_embeddings.npy', f'_clusters_kmeans.pkl')
    
    cluster_data = {
        'labels': labels, 
        'centroids': clusterer.cluster_centers_,
        'n_clusters': n_clusters,
        'silhouette_score': final_silhouette,
        'auto_tune_method': 'silhouette' if auto_tune else None
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(cluster_data, f)
    
    print(f"Clustered {len(embeddings)} embeddings into {n_clusters} clusters using K-means")
    print(f"Saved to {output_file}")
    
    return cluster_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file")
    parser.add_argument("--clusters", type=int, help="Number of clusters (ignored if auto-tune is used)")
    parser.add_argument("--auto-tune", action='store_true', 
                        help="Automatically determine optimal number of clusters using silhouette analysis")
    parser.add_argument("--max-clusters", type=int, default=1200, 
                        help="Maximum number of clusters to test for auto-tuning")
    parser.add_argument("--min-clusters", type=int, default=1100, 
                        help="Maximum number of clusters to test for auto-tuning")
    parser.add_argument("--step", type=int, default=5,
                        help="Step size for cluster range (e.g., test 2, 7, 12, 17... with step=5)")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    cluster_embeddings(args.embeddings, args.clusters, args.auto_tune, args.max_clusters, args.min_clusters, args.step, args.output) 