import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.decomposition import PCA

def visualize_clusters(embeddings_file, clusters_file):
    # Load data
    embeddings = np.load(embeddings_file)
    with open(clusters_file, 'rb') as f:
        cluster_data = pickle.load(f)
    labels = cluster_data['labels']
    
    # Reduce dimensions using PCA
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Add explained variance to the plot
    explained_var = reducer.explained_variance_ratio_
    plt.text(0.02, 0.98, f'Explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    output_file = clusters_file.replace('.pkl', '_pca_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {output_file}")
    print(f"Total explained variance: {sum(explained_var):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file")
    parser.add_argument("--clusters", required=True, help="Path to clusters .pkl file")
    
    args = parser.parse_args()
    visualize_clusters(args.embeddings, args.clusters) 