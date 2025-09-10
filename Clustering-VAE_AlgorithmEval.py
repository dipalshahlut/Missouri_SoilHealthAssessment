import pandas as pd
import numpy as np
import logging
import os
import sys

# Import utility functions
from vae_model import VAE, vae_loss_function, train_vae, extract_latent_representations
from data_preparation import load_and_prepare_data, scale_features
from plotting_utils import plot_training_loss, plot_silhouette_comparison, plot_latent_space_2d
from clustering_evaluation import evaluate_multiple_algorithms, get_best_clustering_results
from geographic_visualization import merge_with_spatial_data, visualize_clusters_on_map

SEED = 42
np.random.seed(SEED)

# Configuration
BASE_DIR = "/Users/dscqv/Desktop/SHA_copy/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "aggResult")
SHAPEFILE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "shapefiles_with_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SHAPEFILE_OUTPUT_DIR, exist_ok=True)

ANALYSIS_DEPTH = 30
VAE_LATENT_DIM = 2

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main workflow for multi-algorithm clustering evaluation."""
    
    logging.info("--- 1. Data Preparation ---")
    df, data_scaled, cluster_cols_all = load_and_prepare_data(BASE_DIR, ANALYSIS_DEPTH)
    
    logging.info("--- 2. VAE Training ---")
    model, train_losses = train_vae(data_scaled, VAE_LATENT_DIM, epochs=100)
    plot_training_loss(train_losses, OUTPUT_DIR)
    
    logging.info("--- 3. Extracting Latent Representations ---")
    z_mean = extract_latent_representations(model, data_scaled)
    
    logging.info("--- 4. Multi-Algorithm Clustering Evaluation ---")
    k_range = range(2, 21)
    methods = ["KMeans", "Agglomerative", "Birch", "GMM"]
    
    scores, best_k_for, best_score_for = evaluate_multiple_algorithms(
        z_mean, methods, k_range
    )
    
    # Plot comparison
    plot_silhouette_comparison(scores, k_range, methods, OUTPUT_DIR)
    
    logging.info("--- 5. Generating Best Clustering Results ---")
    df_with_clusters = get_best_clustering_results(df, z_mean, best_k_for, methods)
    
    # Save results
    df_with_clusters.to_csv(
        os.path.join(OUTPUT_DIR, "main_with_best_labels_allAlgo.csv"), 
        index=False
    )
    
    logging.info("--- 6. Spatial Visualization ---")
    final_merged_gdf = merge_with_spatial_data(df_with_clusters, BASE_DIR)
    
    # Visualize each algorithm's best result
    for method in methods:
        if best_k_for[method] is not None:
            cluster_col = f"{method}_best{best_k_for[method]}"
            visualize_clusters_on_map(
                final_merged_gdf, 
                cluster_col, 
                best_k_for[method],
                f"k={best_k_for[method]} (VAE-{method}, {ANALYSIS_DEPTH}cm)",
                OUTPUT_DIR
            )
    
    logging.info("--- 7. Latent Space Visualization ---")
    best_method = max(best_score_for, key=best_score_for.get)
    best_k = best_k_for[best_method]
    best_cluster_col = f"{best_method}_best{best_k}"
    
    plot_latent_space_2d(
        z_mean, 
        df_with_clusters[best_cluster_col], 
        best_k,
        f"VAE Latent Space - {best_method} (k={best_k})",
        OUTPUT_DIR,
        f"vae_{best_method.lower()}_latent_2d_k{best_k}.png"
    )
    
    logging.info("Multi-algorithm clustering analysis complete!")

if __name__ == "__main__":
    main()
