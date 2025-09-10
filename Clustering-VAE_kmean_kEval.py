#################################################################################################################################
import os
import sys
# --- Configuration ---
# IMPORTANT: Set this to the exact path of the folder containing the PNG files
FOLDER_PATH = "/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data/aggResult" # <--- CHANGE THIS PATH !!!

# --- !!! MASTER DELETE SWITCH !!! ---
# Set to True to ACTUALLY DELETE files.
# Set to False to only LIST the files that would be deleted (simulation mode).
ENABLE_DELETION = True # <-- SAFER DEFAULT! Change to True only when ready.

# --- Safety Check: Verify Folder Existence ---
if not os.path.isdir(FOLDER_PATH):
    print(f"Error: The specified folder does not exist: {FOLDER_PATH}")
    print("Please correct the FOLDER_PATH variable in the script.")
    sys.exit(1) # Exit the script if the folder isn't found

# --- Find PNG Files ---
files_to_delete = []
print(f"Scanning folder: {FOLDER_PATH}")
try:
    for filename in os.listdir(FOLDER_PATH):
        # Check if the file ends with .png (case-insensitive)
        if filename.lower().endswith(".png"):
            file_path = os.path.join(FOLDER_PATH, filename)
            # Ensure it's actually a file, not a directory named something.png
            if os.path.isfile(file_path):
                files_to_delete.append(file_path)
except OSError as e:
    print(f"Error accessing the folder: {e}")
    sys.exit(1)

# --- Process Based on ENABLE_DELETION Flag ---
if not files_to_delete:
    print("No .png files found in the specified folder.")
elif ENABLE_DELETION:
    # --- Deletion Enabled ---
    print("\n" + "="*30)
    print("!!! WARNING: DELETION IS ENABLED !!!")
    print("="*30)
    print("\nThe following .png files will be PERMANENTLY DELETED:")
    for file_path in files_to_delete:
        print(f"  - {os.path.basename(file_path)}")
    print("-" * 30)
    print(f"TOTAL FILES TO DELETE: {len(files_to_delete)}")
    print("-" * 30)

    # Ask for explicit confirmation (still recommended even with the flag)
    confirm = input("Are you absolutely sure you want to proceed with deletion? (yes/no): ").lower().strip()

    if confirm == 'yes':
        print("\nProceeding with deletion...")
        deleted_count = 0
        error_count = 0
        # --- Perform Deletion ---
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"  Deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except OSError as e:
                print(f"  ERROR deleting {os.path.basename(file_path)}: {e}")
                error_count += 1

        print("-" * 30)
        print(f"Deletion complete. {deleted_count} file(s) successfully deleted.")
        if error_count > 0:
            print(f"{error_count} file(s) could not be deleted due to errors (e.g., permissions).")
    else:
        print("\nDeletion cancelled by user confirmation. No files were deleted.")
else:
    # --- Deletion Disabled (Simulation Mode) ---
    print("\n" + "="*30)
    print("--- Deletion is DISABLED (Simulation Mode) ---")
    print("="*30)
    print("\nThe following .png files WOULD BE deleted if ENABLE_DELETION were True:")
    for file_path in files_to_delete:
        print(f"  - {os.path.basename(file_path)}")
    print("-" * 30)
    print(f"TOTAL FILES FOUND: {len(files_to_delete)}")
    print("\nNo files were actually deleted.")

print("\nScript finished.")

#################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For color mapping
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, PowerTransformer,
    RobustScaler, Normalizer, QuantileTransformer)
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering, DBSCAN, SpectralClustering, MiniBatchKMeans, Birch,MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture # Removed as it's not used
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances
from collections import Counter
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D # For custom legends
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import os
import math
from pandas.plotting import parallel_coordinates
from scipy.stats.mstats import winsorize
from pandas.plotting import scatter_matrix
import skfuzzy as fuzz
# --- Seed Setting (Crucial for Reproducibility) ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Geographic Processing ---
# Assuming geographic_processing functions are defined elsewhere
# Make sure this import works or define the functions directly here
#from clustering import (prepare_data_for_clustering)
try:
    from geographic_processing import (load_and_filter_spatial_data_new,
                                       reproject_and_calculate_area,
                                       save_spatial_data)
    GEOGRAPHIC_PROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: 'geographic_processing' module not found. Geographic visualization will be skipped.")
    GEOGRAPHIC_PROCESSING_AVAILABLE = False
    # Define dummy functions if needed to prevent NameErrors later
    def load_and_filter_spatial_data_new(*args, **kwargs): return None
    def reproject_and_calculate_area(*args, **kwargs): return None
    def save_spatial_data(*args, **kwargs): pass


# --- Configuration ---
AREA_NORM = False       # set False to skip dividing soil vars by area
BASE_DIR = "/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data" #<-- UPDATE YOUR PATH
OUTPUT_DIR = os.path.join(BASE_DIR, "aggResult")
SHAPEFILE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "shapefiles_with_data")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output dir exists
os.makedirs(SHAPEFILE_OUTPUT_DIR, exist_ok=True)

ANALYSIS_DEPTH = 30
depth_suffix = f'_{ANALYSIS_DEPTH}cm'
# Define columns intended for clustering (mukey is identifier, area_ac might be feature or just context)
cluster_cols_base = [f'clay{depth_suffix}',f'sand{depth_suffix}', f'om{depth_suffix}', f'cec{depth_suffix}', f'bd{depth_suffix}',
                     f'ec{depth_suffix}', f'pH{depth_suffix}',  
                     f'ksat{depth_suffix}', f'awc{depth_suffix}'] # f'lep{depth_suffix}', f'clay{depth_suffix}',f'cec{depth_suffix}', 


# Add MnRs_dep and area_ac conditionally or based on strategy
cluster_cols_all = ['area_ac','MnRs_dep'] + cluster_cols_base  # 'area_ac',

# Choose VAE latent dimension (e.g., 3 for 3D plotting focus, 2 otherwise)
VAE_LATENT_DIM = 2# <-- SET VAE DIMENSION HERE
PLOT_3D = (VAE_LATENT_DIM >= 3) # Flag to control 3D plotting

# --- Logging Setup (Optional but good practice) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Data Preparation ---
logging.info("--- 1. Data Preparation ---")
#input_csv_path = os.path.join(BASE_DIR, "aggResult", f'MO_{ANALYSIS_DEPTH}cm_for_clustering_grp1.csv')
input_csv_path = os.path.join(BASE_DIR, "aggResult", f'MO_{ANALYSIS_DEPTH}cm_for_clustering.csv')
#input_csv_path = os.path.join(BASE_DIR, "aggResult", f'MO_{ANALYSIS_DEPTH}cm_for_clustering_county_all.csv')
# input_csv_path = os.path.join(BASE_DIR, "aggResult", 'MO_30cm_for_clustering_mukeyfilter.csv') # Alternative input

if not os.path.exists(input_csv_path):
    logging.error(f"Input file not found: {input_csv_path}")
    sys.exit(1)

try:
    df = pd.read_csv(input_csv_path)
    # Ensure mukey is present and maybe set as index if unique and desired
    if 'mukey' not in df.columns:
        logging.warning("'mukey' column not found in input CSV.")
        # Decide how to handle this - exit or proceed without mukey?
        # For now, proceed, but geographic merge will fail.
    else:
        # Optional: Set mukey as index if it's unique and helps tracking
        # if df['mukey'].is_unique:
        #     df.set_index('mukey', inplace=True)
        # else:
        #     logging.warning("mukey column is not unique, cannot set as index.")
        pass # Keep mukey as a regular column for merging later
except FileNotFoundError:
    logging.error(f"Input file not found: {input_csv_path}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error reading input CSV: {e}")
    sys.exit(1)

print(f"after loading file, {df.shape}")
# Check if necessary columns exist
missing_data_cols = [col for col in cluster_cols_all if col not in df.columns]
if missing_data_cols:
    logging.error(f"Missing required data columns: {missing_data_cols}")
    sys.exit(1)

target_mukeys = [2498901, 2498902, 2500799, 2500800, 2571513, 2571527]#[2498901, 2500799]#
df = df[~df['mukey'].isin(target_mukeys)]
print(df.shape)
# Select potential clustering columns initially
data = df[cluster_cols_all].copy()
print(data.iloc[:,1:].columns)
logging.info(f"Original data shape (potential features): {data.shape}")
print(f"The aggregated 0-30cm variables statistic is : {data.describe()}")


# # Impute missing values (using mean imputation)
# # Consider other strategies like median or KNNImputer if appropriate
imputer = SimpleImputer(strategy='mean')  #KNNImputer(n_neighbors=2)#
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=cluster_cols_all, index=data.index)#[1:]
logging.info("Missing values imputed using mean strategy.")


# --- Scaling ---
# Choose scaler: StandardScaler (sensitive to outliers) or RobustScaler (less sensitive)
scaler = RobustScaler()

# AREA_NORM is False: Use features directly, excluding 'area_ac' as it's often context/weight
logging.info("AREA_NORM is False. Selecting features excluding 'area_ac' for scaling.")
features_to_scale = ['MnRs_dep'] + cluster_cols_base # Exclude 'area_ac'
data_to_scale = data_imputed[features_to_scale]

# Apply the chosen scaler
logging.info(f"Applying {scaler.__class__.__name__} to selected features.")
scaled_values = scaler.fit_transform(data_to_scale)
data_scaled = pd.DataFrame(scaled_values, columns=features_to_scale, index=data_to_scale.index)
data_rs = data_scaled.copy()
# add "_sc" suffix to each scaled feature name
scaled_columns = [f"{col}_sc" for col in features_to_scale]

data_rs = pd.DataFrame(
    scaled_values,
    columns=scaled_columns,
    index=data_to_scale.index
)
print(data_rs.shape)
data_joined = pd.concat([data, data_rs], axis=1)
print(data_joined.columns)
print(data_joined.head())
data_joined.to_csv("/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data/aggResult/shapefiles_with_data/Kmeans_best10/data_joined.csv", index=False)
logging.info(f"Scaled data shape: {data_scaled.shape}")
# Display first few rows of scaled data
# print("Scaled data head:\n", data_scaled.head())

# Convert scaled data to PyTorch tensor
try:
    data_tensor = torch.tensor(data_scaled.values, dtype=torch.float32)
    logging.info("Data prepared, scaled, and converted to tensor.")
except Exception as e:
    logging.error(f"Error converting scaled data to tensor: {e}")
    sys.exit(1)

# --- 2. Build the VAE ---
logging.info(f"--- 2. Building VAE (Latent Dim = {VAE_LATENT_DIM}) ---")
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim1=64, hidden_dim2=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc5 = nn.Linear(hidden_dim1, input_dim)
        # Activation
        self.relu = nn.ReLU() # Could try LeakyReLU etc.

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from N(0, I)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        # No activation on final layer for reconstruction (assuming continuous features)
        # If features were bounded (e.g., sigmoid for [0,1]), apply here.
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# VAE Loss function (ELBO)
def vae_loss_function(x_recon, x, mu, logvar):
    # Reconstruction Loss (Mean Squared Error)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    # KL Divergence Loss (analytical solution for Gaussian)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss # Returns total loss (summed over batch)

input_dim = data_scaled.shape[1]
model = VAE(input_dim, VAE_LATENT_DIM)
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Common learning rate
logging.info("VAE model defined.")
# print(model) # Optional: print model structure

# --- 3. Train the VAE ---
logging.info("--- 3. Training VAE ---")
epochs = 100 # Adjust as needed based on convergence
batch_size = 64 # Common batch size, adjust based on memory
dataset = torch.utils.data.TensorDataset(data_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_losses = []
for epoch in range(epochs):
    model.train() # Set model to training mode
    epoch_loss = 0.0
    # epoch_recon_loss = 0.0
    # epoch_kl_loss = 0.0

    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x_batch)

        # Calculate loss using the VAE loss function
        recon_loss_term = nn.functional.mse_loss(x_recon, x_batch, reduction='sum')
        kl_loss_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss_term + kl_loss_term # Total loss for the batch

        # Check for NaN loss
        if torch.isnan(loss):
             logging.error(f"NaN loss encountered at Epoch {epoch+1}. Stopping training.")
             # save state, print problematic batch info
             sys.exit(1) # Exit if training fails

        loss.backward() # Compute gradients
        optimizer.step() # Update weights

        epoch_loss += loss.item()
        # epoch_recon_loss += recon_loss_term.item()
        # epoch_kl_loss += kl_loss_term.item()

    # Average losses over the entire dataset for reporting
    avg_epoch_loss = epoch_loss / len(dataloader.dataset)
    # avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
    # avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0: # Print every 10 epochs
        logging.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        # logging.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg KL Loss: {avg_kl_loss:.4f}")

logging.info("VAE training finished.")

# Optional: Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Average VAE Loss')
plt.title('VAE Training Loss Curve')
plt.grid(True)
loss_plot_path = os.path.join(OUTPUT_DIR, 'vae_training_loss.png')
plt.savefig(loss_plot_path, dpi=300)
logging.info(f"VAE training loss plot saved to: {loss_plot_path}")
plt.close()

# --- 4. Extract Latent Representations ---
logging.info(f"--- 4. Extracting Latent Representations (Dim = {VAE_LATENT_DIM}) ---")
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculations
    mu, logvar = model.encode(data_tensor)
    # Use the mean (mu) of the latent distribution as the representation
    z_mean = mu.numpy()
    # Could also use z = model.reparameterize(mu, logvar).numpy() if stochasticity desired downstream

if np.isnan(z_mean).any():
    logging.error("NaN values found in latent representations (z_mean). Check VAE training.")
    sys.exit(1)

logging.info(f"Latent representations (z_mean) extracted, shape: {z_mean.shape}")

# Pure-Python Partitioning Around Medoids (PAM) implementation
def pam(X, k, max_iter=100):
    D = pairwise_distances(X)
    n = X.shape[0]
    medoids = np.random.choice(n, k, replace=False)
    clusters = np.argmin(D[:, medoids], axis=1)
    cost = np.sum(D[np.arange(n), medoids[clusters]])
    for _ in range(max_iter):
        best_cost = cost
        best_medoids = medoids.copy()
        for i in range(k):
            for o in range(n):
                if o in medoids:
                    continue
                new_medoids = medoids.copy()
                new_medoids[i] = o
                new_clusters = np.argmin(D[:, new_medoids], axis=1)
                new_cost = np.sum(D[np.arange(n), new_medoids[new_clusters]])
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_medoids = new_medoids.copy()
        if best_cost < cost:
            medoids = best_medoids
            cost = best_cost
            clusters = np.argmin(D[:, medoids], axis=1)
        else:
            break
    return clusters
# --- 5. Robust K-Means Evaluation to Find Optimal K ---
logging.info("\n--- 5. Robust K-Means Evaluation (Multiple Runs) ---")
k_range = range(10,11) #range(4, 21)  # K values to evaluate (e.g., 4 to 21)
num_runs = 25           # Number of random initializations per k
best_k_per_run = []     # Store the best k found in each run (based on silhouette)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize result containers
# ─────────────────────────────────────────────────────────────────────────────
methods = ["KMeans",
   "Agglomerative",
   "Birch",
   "GMM"
]
methods = [m for m in methods if m is not None]
scores = {method: [] for method in methods}

logging.info(f"Evaluating K-Means for k in {list(k_range)} across {num_runs} runs using Silhouette Score...")
logging.info(f"Using latent space data (z_mean) of shape: {z_mean.shape}")

if z_mean.shape[0] < max(k_range):
     logging.warning(f"Number of samples ({z_mean.shape[0]}) is less than the maximum k ({max(k_range)}).")
     logging.warning("Some k values might result in clusters with very few members or errors.")


for k in k_range:

#  # K-Means
    try:
        km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(z_mean)
        scores["KMeans"].append(silhouette_score(z_mean, km_labels))
    except Exception as e:
        print(f"KMeans failed at k={k}: {e}")
        scores["KMeans"].append(np.nan)

    # Agglomerative
    try:
        ag_labels = AgglomerativeClustering(n_clusters=k).fit_predict(z_mean)
        scores["Agglomerative"].append(silhouette_score(z_mean, ag_labels))
    except Exception as e:
        print(f"Agglomerative failed at k={k}: {e}")
        scores["Agglomerative"].append(np.nan)

    # Birch
    birch_labels = Birch(n_clusters=k).fit_predict(z_mean)
    scores["Birch"].append(silhouette_score(z_mean, birch_labels))

    # Gaussian Mixture
    try:
        gm = GaussianMixture(n_components=k, random_state=42)
        gm_labels = gm.fit_predict(z_mean)
        scores["GMM"].append(silhouette_score(z_mean, gm_labels))
    except Exception as e:
        print(f"GMM failed at k={k}: {e}")
        scores["GMM"].append(np.nan)
    
    
# Convert `scores` to a DataFrame indexed by k
df_scores = pd.DataFrame(scores, index=list(k_range))
df_scores.index.name = "k"

# ───  Identify best‐k for each method ───────────────────────────────────
best_k_for = {}       # e.g. { "KMeans": 3, "Agglomerative": 4, ... }
best_score_for = {}   # e.g. { "KMeans": 0.52, "Agglomerative": 0.49, ... }

for method in methods:
    # **Filter to only those rows where k > 4 before taking idxmax.**
    df_filtered = df_scores.loc[df_scores.index > 4, method]

    # If all values are NaN (or there is no k > 4), handle gracefully:
    if df_filtered.dropna().empty:
        logging.warning(f"No valid silhouette scores for {method} when k > 4. Setting best_k to None.")
        best_k_for[method] = None
        best_score_for[method] = np.nan
    else:
        best_k = df_filtered.idxmax()       # returns the k (> 4) with highest silhouette
        best_score = df_filtered.max()      # the corresponding silhouette score
        best_k_for[method] = best_k
        best_score_for[method] = best_score
    # best_k = df_scores[method].idxmax()        # k value with highest silhouette
    # best_score = df_scores[method].max()       # corresponding silhouette score
    # best_k_for[method] = best_k
    # best_score_for[method] = best_score

# ─────────────────────────────────────────────────────────────────────────────
# Verify that all methods have the correct number of scores
# ─────────────────────────────────────────────────────────────────────────────
for method in methods:
    print(f"{method}: {len(scores[method])} scores (expected {len(k_range)})")

# ─────────────────────────────────────────────────────────────────────────────
# Run each algorithm again using its best k, store labels
# ─────────────────────────────────────────────────────────────────────────────
for method in methods:
    k_opt = best_k_for[method]
    col_name = f"{method}_best{k_opt}"  # e.g. "KMeans_best13"

    if method == "KMeans":
        model = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
        labels = model.fit_predict(z_mean)

    elif method == "MiniBatchKMeans":
        model = MiniBatchKMeans(n_clusters=k_opt, random_state=42)
        labels = model.fit_predict(z_mean)

    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k_opt)
        labels = model.fit_predict(z_mean)

    elif method == "Birch":
        model = Birch(n_clusters=k_opt)
        labels = model.fit_predict(z_mean)

    elif method == "GMM":
        model = GaussianMixture(n_components=k_opt, random_state=42)
        labels = model.fit_predict(z_mean)

    elif method == "FuzzyCMeans":
        cntr_k, u_k, _, _, _, _, _  = fuzz.cluster.cmeans(z_mean.T, c=k_opt, m=2.0, error=0.005, maxiter=1000)
        fuzzy_labels = np.argmax(u_k, axis=0)

    else:
        # If you ever add another method, implement it here
        raise ValueError(f"Unknown method: {method}")

    # Assign the label array to a new column in main_df
    df[col_name] = labels

df.to_csv("/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data/aggResult/main_with_best_labels_allAlgo_0821.csv", index=False)
# unique_vals = sorted(df['KMeans_best13'].unique())
# print(f"Number of unique clusters presnt in column KMeans_best13 are: {unique_vals}")
# nan_count_A = df['KMeans_best13'].isna().sum()
# print(f"Number of NaNs in column KMeans_best13: {nan_count_A}")
# ─────────────────────────────────────────────────────────────────────────────
# Plot only those with correct lengths
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for method in methods:
    if len(scores[method]) == len(k_range):
        plt.plot(k_range, scores[method], marker='o', label=method)
    else:
        print(f"Skipping {method} in plot due to mismatched score length.")

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Clustering Algorithm Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data/aggResult/clustering_silhouette_main_with_best_labels_allAlgo_0609.png', dpi=300, bbox_inches='tight')

# ─────────────────────────────────────────────────────────────────────────────
# Save Final DataFrame with Cluster Labels (Using BEST k cluster results) 
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:5070"
MU_POLY_PATH = os.path.join(BASE_DIR, "mupoly.shp")
mo_mu_shp_initial = load_and_filter_spatial_data_new( MU_POLY_PATH)     
# Reproject and calculate area
mo_mu_shp = reproject_and_calculate_area(mo_mu_shp_initial, TARGET_CRS)
# Pre-merge Checks & Merge Logic
if 'mukey' not in df.columns: print("Cannot merge: 'mukey' missing in the data.")
elif 'mukey' not in mo_mu_shp.columns and 'MUKEY' not in mo_mu_shp.columns: print("Cannot merge: 'mukey'/'MUKEY' missing in mo_mu_shp.")
else:
    df['mukey'] = df['mukey'].astype(str)
    mo_mu_shp_copy = mo_mu_shp.copy()
    if 'MUKEY' in mo_mu_shp_copy.columns and 'mukey' not in mo_mu_shp_copy.columns:
        mo_mu_shp_copy.rename(columns={'MUKEY': 'mukey'}, inplace=True)
    mo_mu_shp_copy['mukey'] = mo_mu_shp_copy['mukey'].astype(str)
    final_merged_gdf = mo_mu_shp_copy.merge(df, on='mukey', how='left')
    print(f"Final merge complete. Result shape: {final_merged_gdf.shape}")
    print(f"Final merge complete. Result columns name are: {final_merged_gdf.columns}")

# ─────────────────────────────────────────────────────────────────────────────
#  # --- Generate 2D Plot for Final K in Latent Space --- with Cluster Labels (Using BEST k cluster results) 
# ─────────────────────────────────────────────────────────────────────────────
# --- Add final labels to DataFrame and Print Counts ---
best_k_col_name = col_name#"KMeans_best13" #"Birch_best5" #

# --- Generate 2D Plot for Final K in Latent Space ---
print(f"Generating 2D Latent Space Plot for Final k={best_k_col_name}")
kmeans_plot_file_2d_final = os.path.join(OUTPUT_DIR, f'vae_kmeans_latent_2d_final_k{best_k_col_name}.png')
fig_km2d_final, ax_km2d_final = plt.subplots(figsize=(12, 9))
cmap = plt.get_cmap('viridis', best_k) # Use a perceptually uniform colormap
model_final = KMeans(n_clusters=13, random_state=42, n_init=10)
#model_final = Birch(n_clusters=5)
labels_final = model_final.fit_predict(z_mean)
final_silhouette_score = silhouette_score(z_mean, labels_final) # Score for the final clustering
scatter = ax_km2d_final.scatter(z_mean[:, 0], z_mean[:, 1], c=labels_final, cmap=cmap, s=20, alpha=0.7,  picker=True)
        # --- Add Interactivity ---
# Store the current annotation object to remove it later
current_annotation = [None] # Use a list to allow modification within the function

def onclick(event):
    # Ignore clicks outside the axes
    if not event.inaxes == ax_km2d_final:
        return

    # Find the data point closest to the click
    click_x, click_y = event.xdata, event.ydata
    distances_sq = (z_mean[:, 0] - click_x)**2 + (z_mean[:, 1] - click_y)**2
    closest_point_idx = np.argmin(distances_sq)

    # Get the data coordinates of the closest point
    point_x, point_y = z_mean[closest_point_idx, 0], z_mean[closest_point_idx, 1]

    # --- Retrieve Original Identifier and Cluster ---
    try:
        # Option 1: Use DataFrame index (if it aligns with z_mean rows)
        original_identifier = df.index[closest_point_idx]
        identifier_type = "Index"
        # Option 2: Use 'mukey' column (if it exists and aligns)
        if 'mukey' in df.columns:
            original_identifier = df['mukey'].iloc[closest_point_idx]
            identifier_type = "mukey"

    except IndexError:
        original_identifier = f"Row {closest_point_idx}" # Fallback if index/mukey retrieval fails
        identifier_type = "Row Num"
    except Exception as e:
            original_identifier = f"Error ({e})"
            identifier_type = "Error"


    cluster_label = labels_final[closest_point_idx]
    info_text = f"{identifier_type}: {original_identifier}\nCluster: {cluster_label}"

    # Print to console
    print(f"Clicked near point {closest_point_idx} -> {identifier_type}: {original_identifier}, Cluster: {cluster_label}")

    # --- Add/Update Annotation ---
    # Remove the previous annotation if it exists
    if current_annotation[0] is not None:
        current_annotation[0].remove()
        current_annotation[0] = None

    # Add new annotation
    annotation = ax_km2d_final.annotate(
        info_text,
        xy=(point_x, point_y), # Point location
        xytext=(15, 15), # Offset text position
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='black')
    )
    current_annotation[0] = annotation # Store the new annotation object
    fig_km2d_final.canvas.draw_idle() # Redraw the figure to show the annotation

# Connect the click event to the onclick function
cid = fig_km2d_final.canvas.mpl_connect('button_press_event', onclick)
# --- End Interactivity ---

# Create a colorbar
cbar = plt.colorbar(scatter, ax=ax_km2d_final)
cbar.set_label(f'Number of Clusters (k={best_k})')
# Optional: Add cluster centers to the plot
# centers_2d = kmeans_final.cluster_centers_[:, :2] # Assuming VAE_LATENT_DIM >= 2
# ax_km2d_final.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', s=150, c='red', edgecolor='black', label='Cluster Centers')

ax_km2d_final.set_title(f"Final {best_k_col_name} algorithm Clusters (k={best_k}, Silhouette={final_silhouette_score:.3f}) - VAE Latent Space (2D)")
ax_km2d_final.set_xlabel('Latent Dimension 1')
ax_km2d_final.set_ylabel('Latent Dimension 2')
ax_km2d_final.grid(True, linestyle='--', alpha=0.6)
# ax_km2d_final.legend() # Add legend if plotting centers

plt.tight_layout()
plt.savefig(kmeans_plot_file_2d_final, dpi=300)
logging.info(f"Saved Final 2D latent space plot (k={best_k}) to: {kmeans_plot_file_2d_final}")
#plt.close(fig_km2d_final)
# --- End 2D Plot ---

# ─────────────────────────────────────────────────────────────────────────────
# Summary Statistics Plot (Using BEST k cluster results) ---
# ─────────────────────────────────────────────────────────────────────────────
# --- Summary Statistics Plot (Scaled Variables vs Cluster using BEST k) ---
print(f"\n---  Plotting Summary Statistics for Each Scaled Variable Across Clusters (k={best_k}) ---")

if best_k_col_name not in df.columns:
    logging.error(f"Cluster column '{best_k_col_name}' not found in DataFrame. Skipping summary statistics plot.")
else:
    try:
        # Combine cluster labels with the SCALED data used for VAE input
        df_plot = pd.concat([df[best_k_col_name], data_scaled], axis=1)

        variables_to_plot = list(data_scaled.columns)

        # Drop rows where cluster label might be missing (shouldn't happen if added correctly)
        df_plot.dropna(subset=[best_k_col_name], inplace=True)
        df_plot[best_k_col_name] = df_plot[best_k_col_name].astype(int) # Ensure integer type for plotting

        ###########################################################################################################################
        order = (df_plot.assign(cluster=best_k_col_name).groupby('cluster')['clay_30cm'].median().sort_values().index)
        df_p = data_scaled.assign(cluster=best_k_col_name).melt('cluster')
        g = sns.catplot(data=df_p, x='cluster', y='value', col='variable',
            kind='box', col_wrap=4, sharey=False, order=order,
            height=3, aspect=1)
        g.set_titles("{col_name}")
        g.set_axis_labels("Cluster", "Scaled value")
        plt.tight_layout()

        ###########################################################################################################################
        print(df_plot.columns)
        print(df_plot.head())
        if df_plot.empty:
                logging.warning("No data remaining for summary plot after handling NAs.")
        else:
            logging.info(f"Plotting summary stats for {len(variables_to_plot)} scaled variables.")
            num_vars = len(variables_to_plot)
            ncols = min(4, num_vars) # Max 4 plots per row
            nrows = (num_vars + ncols - 1) // ncols

            fig_summary, axes_summary = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), squeeze=False)
            axes_flat = axes_summary.flatten()

            for i, var in enumerate(variables_to_plot):
                ax = axes_flat[i]
                if var in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[var]):
                    sns.boxplot(
                        data=df_plot,
                        x=best_k_col_name,
                        y=var,
                        palette='viridis', # Use the same palette as PCA/latent plots
                        showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"},
                        ax=ax,
                        # Reduce whisker length if needed: whis=1.0
                    )

                    # Global mean of the scaled variable (should be ~0 if StandardScaler used properly)
                    global_mean = df_plot[var].mean()
                    ax.axhline(global_mean, color='red', linestyle='--', linewidth=1.5, label=f'Global Mean ({global_mean:.2f})')

                    ax.set_title(var, fontsize=10)
                    ax.set_ylabel('Value (Scaled)') if i % ncols == 0 else ax.set_ylabel('')
                    ax.set_xlabel(f'Cluster (k={best_k})')
                    # Rotate ticks only if many clusters
                    if best_k > 10:
                            ax.tick_params(axis='x', rotation=45, labelsize=8)
                    else:
                            ax.tick_params(axis='x', labelsize=9)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.legend(fontsize='x-small')
                else:
                        # Handle cases where a variable might be missing or non-numeric
                        ax.text(0.5, 0.5, f'{var}\n(Not Plotted)', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(var, fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])


            # Hide unused subplots
            for j in range(num_vars, len(axes_flat)):
                fig_summary.delaxes(axes_flat[j])

            plt.suptitle(f'Box Plots of Scaled Variables Across Clusters (k={best_k_col_name})\n(VAE Input Features)', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

            save_path = os.path.join(OUTPUT_DIR, f'boxplot_scaled_variables_by_cluster_k{best_k_col_name}.png')
            plt.savefig(save_path, dpi=300)
            logging.info(f"Box plot summary saved to: {save_path}")
            plt.close(fig_summary)
    except Exception as e:
            logging.error(f"Error during Summary Statistics Plotting (Section 9): {e}")
            if 'fig_summary' in locals(): plt.close(fig_summary)

    if df_plot.empty:
            logging.warning("No data remaining for summary plot after handling NAs.")
    else:
        logging.info(f"Plotting summary stats for {len(variables_to_plot)} scaled variables.")
        custom_colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
        df_unscaled = df[variables_to_plot].copy()
        df_unscaled = pd.concat([df_unscaled, df_plot[best_k_col_name]], axis=1)
        print(df_unscaled.columns)
        df_unscaled[best_k_col_name] = df_unscaled[best_k_col_name].astype(int) + 1
        # Shift cluster labels to 1-based
        #df_plot[best_k_col_name] = df_plot[best_k_col_name].astype(int) + 1

        # Ensure cluster labels appear in sorted numeric order
        order = list(range(1, best_k + 1))
        #order = list(pd.unique(df_plot[best_k_col_name]))
        pos_by_cat = {cat: j for j, cat in enumerate(order)}  # category -> x position

        for i, var in enumerate(df_unscaled):#(variables_to_plot):
            #if var in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[var]):
            if var in df_unscaled.columns and pd.api.types.is_numeric_dtype(df_unscaled[var]):
                fig, ax = plt.subplots(figsize=(6, 4)) #(7.5, 4.5))#

                sns.boxplot(
                    data=df_unscaled,
                    x=best_k_col_name,
                    y=var,
                    order=order,
                    palette=custom_colors,
                    showmeans=False,
                    meanline=False,
                    ax=ax,
                )

                global_mean = float(df_unscaled[var].mean())# float(df_plot[var].mean())
                ax.axhline(global_mean, color='red', linestyle='--', linewidth=1.5)

                stats = (
                    # df_plot.groupby(best_k_col_name, observed=True)[var]
                    df_unscaled.groupby(best_k_col_name, observed=True)[var]
                    .agg(mean='mean',
                        q1=lambda s: s.quantile(0.25),
                        med='median',
                        q3=lambda s: s.quantile(0.75))
                    .reindex(order)
                )

                half = 0.28
                for cat, row in stats.iterrows():
                    xpos = pos_by_cat[cat]

                    m = row['mean']
                    if pd.notna(m):
                        ax.hlines(float(m), xpos - half, xpos + half, colors='black', linewidth=2.0)

                    med = row['med']
                    if pd.notna(med):
                        ax.scatter([xpos], [float(med)], s=40, zorder=3, marker='D', facecolor='none', edgecolor='black')

                    q1 = row['q1']
                    q3 = row['q3']
                    if pd.notna(q1):
                        ax.hlines(float(q1), xpos - half, xpos + half, colors='black', linewidth=1.2)
                    if pd.notna(q3):
                        ax.hlines(float(q3), xpos - half, xpos + half, colors='black', linewidth=1.2)

                def set_nice_ticks_left(ax, n=8):
                    # autoscale to data first
                    ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
                    y0, y1 = ax.get_ylim()
                    if y0 == y1:
                        y0 -= 0.5; y1 += 0.5  # avoid zero range

                    raw = (y1 - y0) / (n - 1)
                    mag = 10 ** math.floor(math.log10(abs(raw))) if raw != 0 else 1.0
                    base = min([1, 2, 2.5, 5, 10], key=lambda v: abs(v*mag - raw)) * mag

                    # lock to exactly n ticks: y0n + i*base for i=0..n-1, covering data
                    y0n = math.floor(y0 / base) * base
                    y1n = y0n + base * (n - 1)
                    while y1 > y1n:  # extend if data exceeds top
                        y1n += base

                    ax.set_ylim(y0n, y1n)
                    ax.yaxis.set_major_locator(mticker.MultipleLocator(base))
                    ax.yaxis.set_minor_locator(mticker.NullLocator())
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f' if base >= 1 else '%.1f'))
                    ax.tick_params(axis='y', labelsize=8, labelcolor='black')

                def set_nice_ticks_right(ax, ax2, fwd, n=8):
                    # take left limits, transform to unscaled space
                    y0, y1 = ax.get_ylim()
                    u0, u1 = fwd(y0), fwd(y1)
                    if u0 > u1:
                        u0, u1 = u1, u0

                    raw = (u1 - u0) / (n - 1)
                    mag = 10 ** math.floor(math.log10(abs(raw))) if raw != 0 else 1.0
                    base = min([1, 2, 2.5, 5, 10], key=lambda v: abs(v*mag - raw)) * mag

                    # start at a nice multiple and generate exactly n ticks
                    u0n = math.floor(u0 / base) * base
                    ticks = [u0n + i*base for i in range(n)]

                    ax2.set_yticks(ticks)
                    ax2.yaxis.set_minor_locator(mticker.NullLocator())
                    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f' if base >= 1 else '%.1f'))

                # ---- call it ----
                set_nice_ticks_left(ax, n=8)



                # # def nice_step_and_limits(y0, y1, n_ticks=6):
                # #     # raw step
                # #     raw = (y1 - y0) / max(n_ticks - 1, 1)
                # #     mag = 10 ** math.floor(math.log10(raw)) if raw != 0 else 1.0
                # #     nice = min([1, 2, 2.5, 5, 10], key=lambda v: abs(v*mag - raw)) * mag
                # #     # snap limits to multiples of the step
                # #     y0n = math.floor(y0 / nice) * nice
                # #     y1n = math.ceil(y1 / nice) * nice
                # #     return nice, y0n, y1n

                # # # --- Left (scaled) axis: round numbers ---
                # # ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
                # # yl0, yl1 = ax.get_ylim()
                # # step_left, yl0n, yl1n = nice_step_and_limits(yl0, yl1, n_ticks=6)

                # # ax.set_ylim(yl0n, yl1n)
                # # ax.yaxis.set_major_locator(mticker.MultipleLocator(step_left))
                # # # choose integer vs decimal labels based on step size
                # # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f' if step_left >= 1 else '%.2f'))
                # # ax.yaxis.set_minor_locator(mticker.NullLocator())
                # # ax.tick_params(axis='y', labelsize=8, labelcolor='black')
                # # # --- end left-axis block ---

                # # #ax2 = ax.twinx()
                # if var in df_unscaled.columns:
                #     mu = float(df_unscaled[var].mean())
                #     sigma = float(df_unscaled[var].std(ddof=0))
                #     # y0, y1 = ax.get_ylim()
                #     # margin = 0.2 * (y1 - y0)
                    
                #     # if sigma == 0.0:
                #     #     ax2.set_ylim(mu + y0, mu + y1 + margin)
                #     # else:
                #     #     ax2.set_ylim(y0 * sigma + mu, (y1 + margin) * sigma + mu)
                #     if sigma == 0.0:
                #         fwd  = lambda z: z + mu          # scaled -> unscaled
                #         inv  = lambda u: u - mu          # unscaled -> scaled
                #     else:
                #         fwd  = lambda z: z * sigma + mu
                #         inv  = lambda u: (u - mu) / sigma
                #                 # ---- right axis: mirror left ticks in unscaled units ----
                # yticks_left   = ax.get_yticks()          # six nicely spaced ticks we just set
                # ax2           = ax.secondary_yaxis('right', functions=(fwd, inv))

                # # ax2.set_ylim( fwd(ax.get_ylim()[0]), fwd(ax.get_ylim()[1]) )        # same ends
                # # ax2.set_yticks( [fwd(t) for t in yticks_left] )                     # mirror ticks
                # ax2.yaxis.set_major_formatter(
                #         mticker.FormatStrFormatter('%.0f' if sigma >= 1 else '%.1f'))
                # ax2.tick_params(axis='y', labelsize=8, labelcolor='black')
                # # Create a right axis that is tied to the left via the transform
                # # ax2 = ax.secondary_yaxis('right', functions=(fwd, inv))
                # # ---- call it ----
                # #set_nice_ticks_right(ax, ax2, fwd, n=8)
                # # # --- Right (unscaled) axis: round numbers too ---
                # # # get current left limits and convert them to unscaled limits
                # # yl0n, yl1n = ax.get_ylim()
                # # u0 = fwd(yl0n)
                # # u1 = fwd(yl1n)

                # # step_right, u0n, u1n = nice_step_and_limits(min(u0, u1), max(u0, u1), n_ticks=6)

                # # # Put nice round ticks on the right; secondary_yaxis will map them back to left space
                # # ax2.yaxis.set_major_locator(mticker.MultipleLocator(step_right))
                # # ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f' if step_right >= 1 else '%.2f'))
                # # ax2.yaxis.set_minor_locator(mticker.NullLocator())
                # # # ax2.set_ylabel("Value (Unscaled)", fontsize=10, fontweight='bold')

                # # # # Match tick count (equally spaced because left is)
                # # # ax2.yaxis.set_major_locator(LinearLocator(6))
                # # # ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))



  
                # # #ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
                # # # # #Mirror the left Y-ticks onto the right, converting scaled -> unscaled
                # # # def rounded_ticks(ymin, ymax, n_ticks=6):
                # # #     # Step 1: Find step size based on desired number of ticks
                # # #     raw_step = (ymax - ymin) / (n_ticks - 1)
                    
                # # #     # Step 2: Round step to nearest "nice" value (like 0.1, 0.2, 0.25, 0.5, etc.)
                # # #     magnitude = 10**math.floor(math.log10(raw_step))
                # # #     nice_steps = [1, 2, 2.5, 5, 10]
                # # #     step = min(nice_steps, key=lambda x: abs(x*magnitude - raw_step)) * magnitude
                    
                # # #     # Step 3: Snap ymin and ymax to multiples of step
                # # #     ymin_nice = math.floor(ymin / step) * step
                # # #     ymax_nice = math.ceil(ymax / step) * step
                    
                # # #     # Step 4: Generate ticks
                # # #     ticks = np.arange(ymin_nice, ymax_nice + step/2, step)
                # # #     return np.round(ticks, 6)

                # # # # Get current y-limits
                # # # ymin, ymax = ax.get_ylim()
                # # # ticks = rounded_ticks(ymin, ymax, n_ticks=6)
                # # # ax.set_yticks(ticks) 
                # # # ax.tick_params(axis='y', labelsize=8, labelcolor='black')

                # # # # # Set 5 ticks on left Y-axis
                # # # # ax.yaxis.set_major_locator(LinearLocator(numticks=5))
                # # # yticks_scaled = ax.get_yticks()
                # # # # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # 2 decimals
                # # # if sigma == 0.0:
                # # #     yticks_unscaled = mu + yticks_scaled
                # # #     u0, u1 = mu + ax.get_ylim()[0], mu + ax.get_ylim()[1]
                # # # else:
                # # #     yticks_unscaled = yticks_scaled * sigma + mu
                # # #     yl0, yl1 = ax.get_ylim()
                # # #     u0, u1 = yl0 * sigma + mu, yl1 * sigma + mu
                    
                # # # # 3) set right limits to EXACTLY cover the transformed left limits
                # # # #    add an epsilon so the extrema are guaranteed inside the bounds
                # # # eps = (u1 - u0) * 1e-9 if u1 != u0 else 1e-9
                # # # ax2.set_ylim(u0 - eps, u1 + eps)   
                # # # # 4) put exactly 5 ticks on the right, aligned with the left
                # # # ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # 2 decimals
                # # # ax2.set_yticks(yticks_unscaled)

                # # # ymin_ax2, ymax_ax2 = ax2.get_ylim()
                # # # ticks_ax2 = rounded_ticks(ymin_ax2, ymax_ax2, n_ticks=6)
                # # # ax2.set_yticks(ticks_ax2) 
                # # # ax2.tick_params(axis='y', labelsize=8, labelcolor='black')

                ax.set_ylabel("Value (Unscaled)", fontsize=10, fontweight='bold')
                # ax.set_ylabel("Value (Scaled)", fontsize=14, fontweight='bold')
                # ax2.set_ylabel("Value (Unscaled)", fontsize=10, fontweight='bold')
                ax.set_title(var, fontsize=11, fontweight='bold')
                ax.set_xlabel(f'Number of Clusters (k={best_k})', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45 if best_k > 10 else 0, labelsize=8)
                # ax.tick_params(axis='y', labelsize=8, labelcolor='black')

                def auto_place_legend(ax, handles, labels, preferred_loc="upper right",
                                    overlap_threshold=0.05, **kwargs):
                    """
                    1) Prefer preferred_loc if overlap with data is small.
                    2) Else try all inside locations and pick minimal-overlap.
                    3) If still too much overlap, place legend outside (right).
                    """
                    fig = ax.figure
                    fig.canvas.draw()  # ensure renderer

                    # Collect bboxes for visible plotted elements
                    all_bboxes = []
                    renderer = fig.canvas.renderer
                    for artist in (ax.collections + ax.patches + ax.lines):
                        if not artist.get_visible():
                            continue
                        try:
                            bb = artist.get_window_extent(renderer)
                            if bb.width > 0 and bb.height > 0:
                                all_bboxes.append(bb)
                        except Exception:
                            pass

                    if not all_bboxes:
                        # no data → just place preferred
                        return ax.legend(handles=handles, labels=labels, loc=preferred_loc, **kwargs)

                    data_bbox = Bbox.union(all_bboxes)
                    ax_bbox   = ax.get_window_extent()
                    ax_area   = max(ax_bbox.width * ax_bbox.height, 1.0)  # avoid divide-by-zero

                    def overlap_area(bb1, bb2):
                        x0 = max(bb1.x0, bb2.x0); y0 = max(bb1.y0, bb2.y0)
                        x1 = min(bb1.x1, bb2.x1); y1 = min(bb1.y1, bb2.y1)
                        return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))

                    # 1) Try preferred location first
                    leg = ax.legend(handles=handles, labels=labels, loc=preferred_loc, **kwargs)
                    fig.canvas.draw()
                    pref_overlap = overlap_area(leg.get_window_extent(renderer), data_bbox)
                    if (pref_overlap / ax_area) < overlap_threshold:
                        return leg
                    leg.remove()

                    # 2) Test all inside candidates
                    candidates = [
                        "upper right", "upper left", "lower left", "lower right",
                        "center right", "center left", "lower center", "upper center"
                    ]
                    best_loc, best_overlap = None, float("inf")
                    for loc in candidates:
                        leg_tmp = ax.legend(handles=handles, labels=labels, loc=loc, **kwargs)
                        fig.canvas.draw()
                        ov = overlap_area(leg_tmp.get_window_extent(renderer), data_bbox)
                        leg_tmp.remove()
                        if ov < best_overlap:
                            best_overlap, best_loc = ov, loc

                    # 3) If still too overlapping, push outside (right side)
                    if (best_overlap / ax_area) > overlap_threshold:
                        return ax.legend(handles=handles, labels=labels,
                                        loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                        borderaxespad=0.0, **kwargs)

                    return ax.legend(handles=handles, labels=labels, loc=best_loc, **kwargs)


                # # # Legend for each figure
                gm_handle = Line2D([0], [0], color='red', linestyle='--', lw=1.5, label=f'Global Mean (Unscaled {global_mean:.1f})')
                #mean_handle = Line2D([0], [0], color='black', lw=2.0, label='Cluster Mean')
                #med_handle = Line2D([0], [0], marker='D', markersize=6, markerfacecolor='none', markeredgecolor='black', linestyle='None', label='Cluster Median')
                #ax.legend(handles=[gm_handle, mean_handle, med_handle], loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False,borderaxespad=0.2)
                handles = [gm_handle]#, mean_handle, med_handle]
                labels  = [h.get_label() for h in handles]
                auto_place_legend(ax, handles, labels, fontsize=8, frameon=False)
                # ax.legend(handles=[gm_handle, mean_handle, med_handle],
                #         loc='best',
                #         fontsize=8,
                #         frameon=False)
                # Title
                #title_obj = ax.set_title(var, fontsize=11, fontweight='bold', loc='left', pad =10)
                # fig.canvas.draw_idle()
                # renderer = fig.canvas.get_renderer()
                # title_bb = ax.title.get_window_extent(renderer)      # pixels
                # # Convert the title's right edge (x1) into axes coordinates
                # x_after_title, y_title = ax.transAxes.inverted().transform((title_bb.x1, title_bb.y1))
                # ax.legend(handles=[gm_handle, mean_handle, med_handle],
                #     loc='center left',#'upper center',
                #     bbox_to_anchor=(x_after_title + 0.02, 1.0),  #(0.5, 1.15),
                #     ncol=3,           # spread across a row
                #     fontsize=8,
                #     frameon=False,borderaxespad=0.0)
                plt.grid(True,color="#ebebeb", alpha=1.0, linewidth=0.8)  # enable horizontal
                ax.xaxis.grid(False)  # disable vertical
                plt.tight_layout()
                save_path = os.path.join(OUTPUT_DIR, f'{var}_boxplot_cluster_k{best_k}.png')
                plt.savefig(save_path, dpi=300)
                logging.info(f"Saved boxplot for {var} at {save_path}")
                plt.close(fig)
        # custom_colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        #                 "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
        # # Unscaled stats source: ensure these columns exist here
        # df_unscaled = df[variables_to_plot].copy()

        # num_vars = len(variables_to_plot)
        # ncols = 5
        # nrows = int(np.ceil(num_vars / ncols))

        # fig_summary, axes_summary = plt.subplots(nrows=2, ncols=5, figsize=(24, 6)
        #     #nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), squeeze=False
        # )
        # axes_flat = axes_summary.flatten()

        # # Use a fixed category order matching seaborn
        # order = list(pd.unique(df_plot[best_k_col_name]))
        # pos_by_cat = {cat: j for j, cat in enumerate(order)}  # category -> x position

        # for i, var in enumerate(variables_to_plot):
        #     ax = axes_flat[i]

        #     if var in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[var]):

        #         # --- scaled boxplot (left axis) ---
        #         sns.boxplot(
        #             data=df_plot,
        #             x=best_k_col_name,
        #             y=var,
        #             order=order,
        #             palette=custom_colors,#'viridis',
        #             showmeans=False,   # we'll draw our own mean lines
        #             meanline=False,
        #             ax=ax,
        #         )

        #         # --- Global mean (scaled) ---
        #         global_mean = float(df_plot[var].mean())
        #         ax.axhline(global_mean, color='red', linestyle='--', linewidth=1.5)
        #         # Add custom legend entry starting from second plot
        #         if i > 0:
        #             legend_elements = [
        #                 Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Global Mean (Scaled {global_mean:.2f})')
        #             ]
        #             ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)
        #         # ax.text(0.98, 0.95, f"{u'—'*3}Global Mean (Scaled {global_mean:.2f})",
        #         #         transform=ax.transAxes, ha='right', va='top',
        #         #         fontsize=9, color='black')  # transparent text, no bbox

        #         # --- per-cluster stats (scaled), safe to NaNs ---
        #         stats = (
        #             df_plot.groupby(best_k_col_name, observed=True)[var]
        #             .agg(mean='mean',
        #                 q1=lambda s: s.quantile(0.25),
        #                 med='median',
        #                 q3=lambda s: s.quantile(0.75))
        #             .reindex(order)  # ensure rows in same order as boxes
        #         )

        #         half = 0.28  # half-width of short mean/quantile lines

        #         for cat, row in stats.iterrows():
        #             xpos = pos_by_cat[cat]

        #             # mean as short horizontal line (only if not NaN)
        #             m = row['mean']
        #             if pd.notna(m):
        #                 ax.hlines(float(m), xpos - half, xpos + half, colors='black', linewidth=2.0)

        #             # median as diamond marker
        #             med = row['med']
        #             if pd.notna(med):
        #                 ax.scatter([xpos], [float(med)], s=40, zorder=3,
        #                         marker='D', facecolor='none', edgecolor='black')

        #             # Q1 & Q3 short ticks
        #             q1 = row['q1']; q3 = row['q3']
        #             if pd.notna(q1):
        #                 ax.hlines(float(q1), xpos - half, xpos + half, colors='black', linewidth=1.2)
        #             if pd.notna(q3):
        #                 ax.hlines(float(q3), xpos - half, xpos + half, colors='black', linewidth=1.2)

        #         # --- right axis in unscaled units for every subplot ---
        #         ax2 = ax.twinx()
        #         if var in df_unscaled.columns:
        #             mu = float(df_unscaled[var].mean())
        #             sigma = float(df_unscaled[var].std(ddof=0))  # StandardScaler convention
        #             y0, y1 = ax.get_ylim()
        #             margin = 0.15 * (y1 - y0)   # 10% space above data
        #             if sigma == 0.0:
        #                 ax2.set_ylim(mu + y0, mu+y1 + margin)
        #             else:
        #                 ax2.set_ylim(y0 * sigma + mu, (y1+margin) * sigma + mu)
        #         ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        #         ax2.tick_params(axis='y', labelsize=8, labelcolor= 'black')

        #         # Remove all individual y-axis labels
        #         ax.set_ylabel("")
        #         ax2.set_ylabel("")
        #         # titles / ticks
        #         ax.set_title(var, fontsize=10)
        #         ax.set_xlabel('')
        #         #ax.set_xlabel(f'Cluster (k={best_k})')
        #         ax.tick_params(axis='x', rotation=45 if best_k > 10 else 0,
        #                     labelsize=8 if best_k > 10 else 9)
        #         ax.tick_params(axis='y', labelsize=8, labelcolor= 'black')

        #         # legend (only once, transparent frame)
        #         if i == 0:
        #             gm_handle = Line2D([0], [0], color='red', linestyle='--', lw=1.5,
        #                             label=f'Global Mean (Scaled {global_mean:.2f})')
        #             mean_handle = Line2D([0], [0], color='black', lw=2.0,
        #                                 label='Cluster mean (line)')
        #             med_handle = Line2D([0], [0], marker='D', markersize=6,
        #                                 markerfacecolor='none', markeredgecolor='black',
        #                                 linestyle='None', label='Median')
        #             ax.legend(handles=[gm_handle, mean_handle, med_handle],
        #                     loc='upper right', fontsize='x-small', frameon=False)

        #         # # axis labels only at row edges
        #         row = i // ncols
        #         col = i % ncols
        #         if col == 0 :
        #             ax.set_ylabel('Value (Scaled)', fontsize=10, fontweight='bold')
        #             ax2.tick_params(axis='y', labelsize=8, labelcolor='black')
        #         # else:
        #         #     ax.set_ylabel('')
        #          # Y-axis: remove variable name
        #         # ax.set_ylabel("")

        #         # # Add twin axis for unscaled value
        #         # twin_ax = ax.twinx()
        #         # twin_ax.set_ylabel("")
        #         # twin_ax.set_yticks(ax.get_yticks())  # Match ticks   

        #         # if col == ncols - 1:
        #         #     #ax2 = ax.twinx()
        #         #     ax2.set_ylabel('Value (Unscaled)', fontsize=10)


        #     # else:
        #     #     ax.text(0.5, 0.5, f'{var}\n(Not Plotted)', ha='center', va='center',
        #     #             transform=ax.transAxes)
        #     #     ax.set_title(var, fontsize=10)
        #     #     ax.set_xticks([]); ax.set_yticks([])

        # # # Hide any unused axes
        # # for j in range(num_vars, len(axes_flat)):
        # #     fig_summary.delaxes(axes_flat[j])

        # # for j in range(ncols):
        # #     axes_summary[nrows-1, j].set_xlabel('')

        # fig_summary.text(0.02, 0.5, 'Value (Scaled)', va='center', ha='center', rotation='vertical', fontsize=14, fontweight='bold')
        # fig_summary.text(0.98, 0.5, 'Value (Unscaled)', va='center', ha='center', rotation='vertical', fontsize=14, fontweight='bold')
        # # put label only on the middle axis of the bottom row
        # mid_col = ncols // 2
        # axes_summary[nrows-1, mid_col].set_xlabel(f'Cluster (k={best_k})', fontsize=14, fontweight='bold')
        # plt.subplots_adjust(left=0.1, right=0.92, top=0.92, bottom=0.15, wspace=0.35, hspace=0.35)  # more margin for legends
        # plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        # save_path = os.path.join(OUTPUT_DIR, f'boxplot_scaled-unscaled_variables_by_cluster_k{best_k}_vars.png')
        # plt.savefig(save_path, dpi=300)
        # logging.info(f"Box plot summary saved to: {save_path}")
        # plt.close(fig_summary)
# --- End Summary Stats ---
print(df.columns)
# --- group & sum ---
area_by_cluster = (df.groupby(best_k_col_name)['area_ac'].sum().sort_index())     # sort by cluster label

# --- plot ---
plt.figure(figsize=(8, 5))
area_by_cluster.plot(kind='bar', edgecolor='k', alpha=0.7)

plt.xlabel('Cluster Number')
plt.ylabel('Total Area (ac)')
plt.title('Total Area by kMean clusters (k=10)')# Birch Cluster (k=5)')# 
plt.xticks(rotation=0)
plt.tight_layout()
area_path = os.path.join(OUTPUT_DIR, f'cluster_area_cover_plot_{best_k_col_name}.png')
plt.savefig(area_path, dpi=300)
logging.info(f"Cluster area(acre) plot: {area_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Geographic Visualization (Using BEST k cluster results) ---
# ─────────────────────────────────────────────────────────────────────────────

logging.info("Saving merged spatial data...")
final_filename_base = f"MO_{ANALYSIS_DEPTH}cm_clusters_vae_algorithms_merged" # Include k in filename
save_spatial_data(final_merged_gdf, SHAPEFILE_OUTPUT_DIR, final_filename_base) # Use your function
logging.info(f"Merged spatial data saved with base name: {final_filename_base}")


# --- Define the visualization function (adapt as needed) ---
def visualize_clusters_on_map(gdf_merged, cluster_col_name, k_clusters, title_suffix="", output_dir="."):
    """Visualizes clusters on a map."""
    if gdf_merged is None or gdf_merged.empty:
        logging.warning("Merged GeoDataFrame is empty. Cannot visualize map.")
        return
    if cluster_col_name not in gdf_merged.columns:
        logging.error(f"Cluster column '{cluster_col_name}' not found in GeoDataFrame.")
        return

    logging.info(f"Visualizing map for: {cluster_col_name}...")
    gdf_plot = gdf_merged.copy()

    # Handle potential NaN values from left merge (assign a specific category)
    # Use -1 for NaNs, ensure integer type for coloring
    gdf_plot[cluster_col_name] = gdf_plot[cluster_col_name].fillna(-1).astype(int)

    # Create a colormap - use 'tab20' or 'viridis' etc. for categorical data
    # Ensure enough distinct colors for k clusters + NaN category
    cmap = cm.get_cmap('viridis', k_clusters) # Or 'tab20', 'nipy_spectral'
    colors = [cmap(i) for i in range(k_clusters)]
    # Assign colors: cluster 0 gets colors[0], ..., cluster k-1 gets colors[k-1]
    # Assign a distinct color (e.g., grey) for NaN (-1)
    color_mapping = {i: colors[i] for i in range(k_clusters)}
    color_mapping[-1] = (0.7, 0.7, 0.7, 1.0) # Grey for NaN/unmerged

    gdf_plot['plot_color'] = gdf_plot[cluster_col_name].map(color_mapping)

    # Create legend elements manually for clarity
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                                markerfacecolor=colors[i], markersize=8) for i in range(k_clusters)]
    if -1 in gdf_plot[cluster_col_name].unique():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='No Data/Unmerged',
                                        markerfacecolor=color_mapping[-1], markersize=8))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_plot.plot(color=gdf_plot['plot_color'], ax=ax, linewidth=0.1, edgecolor='face') # Thin/no borders
    ax.set_title(f"Map Unit Clusters - {title_suffix}", fontsize=15)
    ax.set_axis_off()

    # Add legend outside the plot area
    ax.legend(handles=legend_elements, title="Clusters", loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_filename = os.path.join(output_dir, f"cluster_map_{cluster_col_name}_vae-kmean.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Cluster map saved to {plot_filename}")
    plt.close(fig)

import re
m = re.search(r'(\d+)$', best_k_col_name)
k_val = int(m.group(1))
best_algo = best_k_col_name.split('_')[0]

# --- Visualize the final cluster result ---
visualize_clusters_on_map(
    gdf_merged=final_merged_gdf,
    cluster_col_name=best_k_col_name,#"{cluster_col_name}",
    k_clusters=k_val,
    title_suffix=f"k={k_val} (VAE-kMean, {ANALYSIS_DEPTH}cm)",
    output_dir=OUTPUT_DIR)
# --- End Geographic Vis ---

# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance via a Supervised Proxy ( Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.inspection import PartialDependenceDisplay


# 1. Extract X (features) and y (cluster labels)
numeric_features = ['MnRs_dep'] + cluster_cols_base 
print(numeric_features)
X = df[numeric_features].values           # shape (n_samples, n_features)
print(df[numeric_features])

y = df[best_k_col_name].values                  # shape (n_samples,)

# 2. (Optional) Standardize if features are on very different scales
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)   # ← should be (1216, 10)
#    df[numeric_features].values, df[best_k_col_name].values,
#     test_size=0.2, random_state=42, stratify=df[best_k_col_name])

# 4. Fit a RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    n_jobs=-1)
rf.fit(X_train, y_train)

# 5. Evaluate accuracy on the held‐out set
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy on predicting clusters: {acc:.3f}")

# 6. Extract feature importances
importances = rf.feature_importances_

feat_importance = sorted(
    zip(numeric_features, importances),
    key=lambda x: x[1], 
    reverse=True)

print("Feature importances (RF) for predicting cluster labels:")
for feat, score in feat_importance[:10]:
    print(f"  {feat}: {score:.3f}")

# Use SHAP (SHapley Additive exPlanations) on the trained

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
# shap_values is a list: one array per cluster.
# shap_values[i][j, k] = impact of feature k on assignment to cluster i for sample j

# 1) Verify that shap_values is a list and inspect each array’s shape
print("Type of shap_values:", type(shap_values))
print("Number of classes (len of shap_values):", len(shap_values))

# for idx, arr in enumerate(shap_values):
#     # Each arr should be a (n_test_samples, n_features) NumPy array
#     print(f"  Class {idx} SHAP array shape: {arr.shape}")

# 2) Verify that X_test has the same shape as each shap_values[i]
print("X_test.shape: {X_test.shape}")
print("Number of feature names:", len(numeric_features))
print(shap_values.shape)

def top_n_features_per_cluster(
    df: pd.DataFrame,
    numeric_features: list[str],
    cluster_col: str,
    shap_values: np.ndarray,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing, for each cluster:
      - the top_n features by mean(|SHAP|)
      - the min/max of each feature within that cluster
    """
    n_classes = shap_values.shape[2]
    records = []
    
    for class_idx in range(n_classes):
        # 1) mean absolute SHAP per feature for this class
        abs_shap = np.abs(shap_values[..., class_idx])  # (n_samples, n_features)
        mean_abs_shap = abs_shap.mean(axis=0)           # (n_features,)
        
        # 2) indices of top_n features
        top_idxs = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        # 3) slice DataFrame for rows in this cluster
        sub = df[df[cluster_col] == class_idx]
        
        # 4) record each top feature
        for rank, idx in enumerate(top_idxs, start=1):
            feat = numeric_features[idx]
            records.append({
                "cluster":      class_idx+1,
                "rank":         rank,
                "feature":      feat,
                "mean_abs_shap": mean_abs_shap[idx],
                "cluster_min":  sub[feat].min(),
                "cluster_max":  sub[feat].max()
            })
    
    return pd.DataFrame(records)

top3_all = top_n_features_per_cluster(
    df, numeric_features, best_k_col_name, shap_values, top_n=3)
print(top3_all)
print(top3_all.dtypes)


# shap_values: (n_samples, n_features, n_clusters)
n_features = shap_values.shape[1]
n_clusters = shap_values.shape[2]

#  Compute mean absolute SHAP across samples for every (feature, cluster)
#    -> shape (n_features, n_clusters)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
# Columns labeled Cluster 1..k (instead of 0..k-1)
col_labels = [f"Cluster {i+1}" for i in range(n_clusters)]

# Build a DataFrame so that rows=features, cols=clusters
heatmap_data = pd.DataFrame(
    mean_abs_shap,
    index=numeric_features,
    columns=col_labels#[f"Cluster {i}" for i in range(n_clusters)]
)

# (Optional) convert to percent of total per cluster
#    so that each column sums to 100%
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Create annotation strings (e.g. "12.3%")
annot = heatmap_pct.round(1).astype(str) + "%"

plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    heatmap_pct,
    annot=annot,
    fmt="",       # one decimal place plus percent sign
    cmap="YlGnBu",
    cbar_kws={'format': '%.0f%%'}  # colorbar ticks in %
)
# Percent formatter for the colorbar (robust)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
cbar.set_label("Mean |SHAP| (%)",
               fontsize=12, fontweight="bold", rotation=270, labelpad=14)
#ax.set_title("Top Features by Mean Absolute SHAP Value per Cluster")
ax.set_xlabel(f'Number of Clusters (k={best_k})', fontsize=14, fontweight='bold')
ax.set_ylabel("Feature", fontsize=14, fontweight='bold')
plt.tight_layout()
featExtr_heatmap_plot = os.path.join(OUTPUT_DIR, f"featExtr_heatmap_{best_k_col_name}.png")
plt.savefig(featExtr_heatmap_plot, dpi=300, bbox_inches='tight')

n_samples, n_features, n_classes = shap_values.shape
# For a global overview of feature importance:
for class_idx in range(n_classes):
    arr = shap_values[:, :, class_idx]
    # Confirm shapes match
    assert arr.shape == X_test.shape  # both should be (1216, 10)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,#arr,
        X_test,
        feature_names=numeric_features,
        plot_type="bar",    # ← bar chart
        show=False)
    plt.title(f"SHAP summary for cluster {best_k}")
    plt.tight_layout()
    SHAP_plot = os.path.join(OUTPUT_DIR, f"SHAPplot_{best_k_col_name}.png")
    plt.savefig(SHAP_plot, dpi=300, bbox_inches='tight')


# Suppose we look at cluster 0’s predicted probability as a function of “clay_30cm”
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    rf,
    X_test,
    features=["MnRs_dep"],
    target=0,  # i.e. cluster label 0
    feature_names=numeric_features,
    ax=ax)
plot_name = os.path.join(OUTPUT_DIR, f"PartialDependenceDisplay_{best_k_col_name}.png")
plt.savefig(plot_name, dpi=300, bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# Additional Descriptive Statistics by Cluster
# ─────────────────────────────────────────────────────────────────────────────
print(df.shape)
print(len(df[best_k_col_name]))
cluster_summary = df[numeric_features].groupby(df[best_k_col_name]).agg(
    ["mean", "median", "std", "min", "max"])
print(f"Each feature’s mean/median/variance differs across clusters: {cluster_summary}")
cluster_summary.to_csv(f'cluster_summary_{best_k_col_name}.csv', index=False)
print(cluster_summary.head())

cluster_summary = pd.read_csv(
    "cluster_summary_KMeans_best10.csv",
    header=[0, 1],           # <— two header rows
    index_col=0              # first column already is the cluster ID 0-9
)
#Remove the overall stats row whose index value is 'KMeans_best10'
cluster_summary = cluster_summary.loc[cluster_summary.index != "KMeans_best10"].copy()

#Give the remaining rows human-friendly labels
cluster_summary.index = [f"Cluster {i}" for i in range(1, len(cluster_summary) + 1)]
#df_raw = pd.read_csv("cluster_summary_KMeans_best10.csv", header=None, nrows=6)
variables = [
    "MnRs_dep",
    "ksat_30cm",
    "cec_30cm",
    "clay_30cm",
    "sand_30cm",
    "om_30cm",
    "bd_30cm",
    "ec_30cm",
    "pH_30cm",
    "awc_30cm",
]
min_df = cluster_summary.xs("min", level=1, axis=1)
max_df = cluster_summary.xs("max", level=1, axis=1)

out = pd.DataFrame(index=cluster_summary.index)

for var in variables:
    if var not in min_df.columns:
        # skip variables that don’t exist in the file
        continue

    out[var] = (
        min_df[var].round(2).astype(str)
        + " – "
        + max_df[var].round(2).astype(str)
    )

print(out)

######################################################################################################

import scipy.stats as stats

results = {}
for feat in numeric_features:
    # Prepare a list of arrays: one array per cluster
    groups = [group[feat].dropna().values for _, group in df[numeric_features].groupby(df[best_k_col_name])]
    f_stat, p_val = stats.f_oneway(*groups)
    results[feat] = p_val
sorted_by_p = sorted(results.items(), key=lambda x: x[1])  # smallest p‐value first
print("Features ordered by ANOVA p‐value (lowest→highest):")
for feat, p in sorted_by_p[:10]:
    print(f"  {feat}: p = {p:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# Cluster‐Centroid Heatmap
# ─────────────────────────────────────────────────────────────────────────────
# 1. Compute centroid table (clusters × features)
centroids = df.groupby(df[best_k_col_name])[numeric_features].mean()

# 2. Z-score normalize centroids (optional, to put all features on same color scale)
centroids_z = (centroids - centroids.mean(axis = 0)) / centroids.std(axis=0, ddof=0)
# 3) Sort clusters, switch to features on Y and clusters on X, relabel clusters 1..k
centroids_plot = centroids_z.sort_index().T              # rows=features, cols=clusters
centroids_plot.columns = [str(int(c) + 1) for c in centroids_plot.columns]  # 1..k labels
# 3. Plot heatmap
plt.figure(figsize=(10, 6))
hm = sns.heatmap(
    centroids_plot, 
    cmap="YlGnBu", 
    center=0, 
    linewidths=0.5, #
    annot=True, 
    fmt=".2f",
    cbar=True,
    cbar_kws={"label": "Z-score of feature mean"}
)
# Labels & title
ax.set_xlabel(f"Cluster (1…{centroids_plot.shape[1]})", fontsize=14, fontweight="bold")
ax.set_ylabel("Feature",                                fontsize=14, fontweight="bold")
#ax.set_title("Cluster Centroids (z-score normalized)",  fontsize=16, fontweight="bold")
# Robust colorbar access (avoid ax.collections[0].colorbar)
cbar = hm.collections[0].colorbar
cbar.set_label("Z-score of feature mean",   # same text
               fontsize=14, fontweight="bold")   # styling here
cbar.ax.tick_params(labelsize=12)           # tick label size (optional)
cbar_ax = ax.figure.axes[-1]             # colorbar axis is added as the last axes
cbar_ax.tick_params(labelsize=11)
cbar_ax.yaxis.label.set_size(12)
cbar_ax.yaxis.label.set_weight("bold")

#cbar_ax.set_ylabel("Z-score of feature mean",fontsize=12, fontweight="bold", rotation=270, labelpad=14)
# Ticks tidy
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#plt.title("Cluster Centroids (z-score normalized)")
plt.ylabel("Feature", fontsize=14, fontweight="bold")
plt.xlabel(f'Number of Clusters (k={best_k})', fontsize=14, fontweight="bold")
plt.tight_layout()
plot_centroid = os.path.join(OUTPUT_DIR, f"CentroidHeatmap_{best_k_col_name}.png")
plt.savefig(plot_centroid, dpi=300, bbox_inches='tight')

# ─────────────────────────────────────────────────────────────────────────────
# Parallel‐Coordinates Plot
# ─────────────────────────────────────────────────────────────────────────────

# 1. Centroids DataFrame with a “cluster” column
centroids = df.groupby(df[best_k_col_name])[numeric_features].mean().reset_index()

plt.figure(figsize=(10, 5))
parallel_coordinates(
    centroids, 
    class_column=best_k_col_name, 
    cols=numeric_features, 
    color=plt.cm.tab10.colors
)
plt.xticks(rotation=45, ha="right")
plt.title("Parallel Coordinates of Cluster Centroids")
plt.ylabel("Feature mean")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
parallel_plot = os.path.join(OUTPUT_DIR, f"ParallelCoordPlot_{best_k_col_name}.png")
plt.savefig(parallel_plot, dpi=300, bbox_inches='tight')
# ─────────────────────────────────────────────────────────────────────────────
# Radar (Spider) Chart
# ─────────────────────────────────────────────────────────────────────────────
# Choose a subset of features (e.g. top 6)
plot_feats = top3_all['feature'].unique()[:6]
angles = np.linspace(0, 2*np.pi, len(plot_feats), endpoint=False).tolist()
angles += angles[:1]  # close the circle
#
centroids = df.groupby(df[best_k_col_name])[plot_feats].mean()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for idx, row in centroids.iterrows():
    values = row.tolist()
    values += values[:1]  # close the loop
    ax.plot(angles, values, label=f"Cluster {idx}")
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(plot_feats)
ax.set_yticklabels([])
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.title("Radar Chart of Cluster Profiles")
Radar_plot = os.path.join(OUTPUT_DIR, f"RadarPlot_{best_k_col_name}.png")
plt.savefig(Radar_plot, dpi=300, bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# PCA Variance Analysis (on ORIGINAL SCALED DATA)
# ─────────────────────────────────────────────────────────────────────────────
# This section runs independently of k, analyzing the variance explained by PCA
# on the input features that went into the VAE.
logging.info("\n---  PCA Variance Analysis (on Original Scaled Data) ---")
print(data_rs.shape)
# Check if data_rs exists and has numeric data
if 'data_rs' not in locals() or not isinstance(data_rs, pd.DataFrame) or data_rs.empty:
    logging.warning("Scaled data ('data_scaled') not available or empty. Skipping PCA Variance Analysis.")
else:
    try:
        # Select only numeric columns just in case
        data_scaled_numeric = data_rs.select_dtypes(include=np.number)

        if data_scaled_numeric.empty:
            logging.warning("No numeric columns found in 'data_scaled'. Skipping PCA Variance Analysis.")
        else:
            if data_scaled_numeric.shape[1] < data_scaled.shape[1]:
                logging.warning("Non-numeric columns excluded from PCA Variance Analysis.")

            n_features = data_scaled_numeric.shape[1]
            pca_full = PCA(n_components=n_features)
            pca_full.fit(data_scaled_numeric.values)


            features_name = data.iloc[:,1:].columns.tolist()
    
            pc_labels = [f"PC{i+1}" for i in range(len(features_name))]
            loadings = pca_full.components_.T                       # shape (n_features, 2)
            loadings_df = pd.DataFrame(loadings,
                                    index=features_name,
                                    columns=pc_labels)
            logging.info("Feature → PC loadings:\n%s", loadings_df)

            # Optional: save to CSV so you can open it in Excel, etc.
            loadings_df.to_csv(os.path.join(OUTPUT_DIR, 'pca_feature_loadings.csv'))
            logging.info("Saved PCA loadings to pca_feature_loadings.csv")

            # --- Log top 3 loadings per PC ---
            top_n = 3
            top_loadings = {}
            for pc in pc_labels:
                # sort by absolute loading, descending
                sorted_abs = loadings_df[pc].abs().sort_values(ascending=False)
                top_feats = sorted_abs.head(top_n).index.tolist()
                top_vals  = loadings_df.loc[top_feats, pc].round(4).tolist()
                top_loadings[pc] = list(zip(top_feats, top_vals))

            logging.info("Top %d feature loadings per PC:", top_n)
            for pc, feats in top_loadings.items():
                formatted = ", ".join(f"{f} ({v:+.4f})" for f, v in feats)
                logging.info(f"  {pc}: {formatted}")

            # Optional: save to CSV
            top_df = (
                pd.DataFrame({
                    'PC':  [pc for pc in pc_labels for _ in range(top_n)],
                    'Feature': [f for feats in top_loadings.values() for f, _ in feats],
                    'Loading': [v for feats in top_loadings.values() for _, v in feats]
                })
            )
            top_df_T = top_df.T
            top_df_T.to_csv(os.path.join(OUTPUT_DIR, 'pca_top3_loadings_per_PC.csv'),
                        index=False)
            
            # create a pivoted view: rows = rank (1st,2nd,3rd), columns = PC, values = "Feature (loading)"
            # first add a rank column so you know 1st vs 2nd vs 3rd
            top_df['Rank'] = top_df.groupby('PC').cumcount() + 1

            pivot_df = top_df.pivot(index='Rank',
                                    columns='PC',
                                    values='Feature').add_prefix('Feat_')

            pivot_vals = top_df.pivot(index='Rank',
                                    columns='PC',
                                    values='Loading').add_prefix('Load_')

            # Combine features and their loadings in one wide table:
            wide_df = pd.concat([pivot_df, pivot_vals], axis=1)
            print(wide_df)
            # Optionally sort columns so Feat_PC1, Load_PC1, Feat_PC2, Load_PC2, …
            # wide_df = wide_df.reindex(
            #     columns=sorted(wide_df.columns, key=lambda x: (int(x.split('_')[2]), x.split('_')[0]))
            # )

            wide_df = wide_df.reindex(columns=sorted(wide_df.columns,key=lambda x: (int(x.split('PC')[1]) if 'PC' in x else float('inf'),
                       x.split('_')[0])))

            print(wide_df)

            # Find the dominant feature for each PC
            dominant = loadings_df.abs().idxmax()       # Series: index=PC labels, value=feature
            dominant_vars = dominant.values             # e.g. ['clay_30cm','awc_30cm', …]

            # Get the variance explained by each PC (in the same order)
            explained = pca_full.explained_variance_ratio_
            # Build x‐labels as "PCi\nfeature"
            scree_labels = [f"{pc_labels[i]}\n{dominant_vars[i]}"
                            for i in range(n_features)]

            # Plot Scree with Dominant‐feature labels
            fig, ax = plt.subplots(figsize=(10,4))
            ax.bar(np.arange(1, n_features+1), explained, edgecolor='k')
            ax.set_xticks(np.arange(1, n_features+1))
            ax.set_xticklabels(scree_labels, rotation=45, ha='right')
            ax.set_xlabel('Dominant Feature (per Principal Component)')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Scree Plot (Labeled by Dominant Variable)')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            # Save
            scree_path = os.path.join(OUTPUT_DIR, 'pca_scree_by_dominant_feature_vae-kmean.png')
            plt.savefig(scree_path, dpi=300)
            plt.close(fig)

            explained_variance = pca_full.explained_variance_
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            pca_variance_df = pd.DataFrame({
                'Component': range(1, n_features + 1),
                'Explained Variance': explained_variance,
                'Explained Variance Ratio': explained_variance_ratio,
                'Cumulative Variance Ratio': cumulative_variance
            })

            logging.info("Explained Variance by Principal Component (on Original Scaled Data):")
            print(pca_variance_df.round(4).to_string(index=False))

            variance_csv = os.path.join(OUTPUT_DIR, 'pca_variance_analysis_original_data.csv')
            pca_variance_df.to_csv(variance_csv, index=False)
            logging.info(f"Saved PCA variance table to: {variance_csv}")

            # Find components needed for a threshold (e.g., 95%)
            threshold = 0.95
            n_thresh_indices = np.where(cumulative_variance >= threshold)[0]
            if len(n_thresh_indices) > 0:
                 n_thresh = n_thresh_indices[0] + 1
                 var_at_thresh = cumulative_variance[n_thresh - 1]
                 logging.info(f"Components needed for >= {threshold * 100:.0f}% variance: {n_thresh} (explains {var_at_thresh * 100:.2f}%)")
            else:
                 n_thresh = n_features # Threshold not reached
                 var_at_thresh = cumulative_variance[-1]
                 logging.warning(f"{threshold * 100:.0f}% variance threshold not reached. Max explained: {var_at_thresh * 100:.2f}% with {n_features} components.")

            # Plot variance
            fig_var, (ax1_var, ax2_var) = plt.subplots(1, 2, figsize=(14, 6))

            # Individual variance plot
            ax1_var.bar(pca_variance_df['Component'], pca_variance_df['Explained Variance Ratio'], alpha=0.8, label='Individual')
            ax1_var.set_xlabel('Principal Component')
            ax1_var.set_ylabel('Explained Variance Ratio')
            ax1_var.set_title('Individual Component Variance')
            tick_step = max(1, n_features // 10) # Adjust x-axis ticks based on number of features
            ax1_var.set_xticks(np.arange(1, n_features + 1, tick_step))
            ax1_var.grid(axis='y', linestyle='--', alpha=0.7)

            # Cumulative variance plot
            ax2_var.plot(pca_variance_df['Component'], pca_variance_df['Cumulative Variance Ratio'], marker='o', linestyle='-', label='Cumulative')
            ax2_var.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold * 100:.0f}% Threshold')
            if n_thresh <= n_features: # Only plot vertical line if threshold is met or n_features is the limit
                ax2_var.axvline(x=n_thresh, color='g', linestyle=':', label=f'{n_thresh} PCs for ~{var_at_thresh*100:.0f}%')
            ax2_var.set_xlabel('Principal Component')
            ax2_var.set_ylabel('Cumulative Variance Ratio')
            ax2_var.set_title('Cumulative Explained Variance')
            ax2_var.set_xticks(np.arange(1, n_features + 1, tick_step))
            ax2_var.grid(True)
            ax2_var.legend(loc='center right') # Adjust legend position

            fig_var.suptitle('PCA Explained Variance Analysis (on Original Scaled Data)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            variance_plot = os.path.join(OUTPUT_DIR, 'pca_variance_plot_original_data.png')
            fig_var.savefig(variance_plot, dpi=300)
            logging.info(f"Saved PCA variance plot to: {variance_plot}")
            plt.close(fig_var)

    except Exception as e:
        logging.error(f"Error during PCA Variance Analysis (Section 8): {e}")
        if 'fig_var' in locals(): plt.close(fig_var)
# --- End PCA Variance ---


#################################################################################################################################
    # ─────────────────────────────────────────────────────────────────────────────
    # Optional Fuzzy C-Means import
    # ─────────────────────────────────────────────────────────────────────────────
    try:
        import skfuzzy as fuzz
        def fuzzy_labels(data, k):
            cntr, u, *_ = fuzz.cluster.cmeans(data.T, c=k, m=2.0, error=0.005, maxiter=1000)
            return np.argmax(u, axis=0)
        HAS_FUZZY = True
    except ImportError:
        HAS_FUZZY = False
        print("skfuzzy not installed, skipping FuzzyCMeans")

    # ─────────────────────────────────────────────────────────────────────────────
    # Cluster methods mapping
    # ─────────────────────────────────────────────────────────────────────────────
    methods = {
        "KMeans":          lambda data, k: KMeans(n_clusters=k, random_state=42).fit_predict(data),
        "MiniBatchKMeans": lambda data, k: MiniBatchKMeans(n_clusters=k, random_state=42).fit_predict(data),
        "Agglomerative":   lambda data, k: AgglomerativeClustering(n_clusters=k).fit_predict(data),
        "Birch":           lambda data, k: Birch(n_clusters=k).fit_predict(data),
        "GMM":             lambda data, k: GaussianMixture(n_components=k, random_state=42).fit_predict(data),
    }
    if HAS_FUZZY:
        methods["FuzzyCMeans"] = fuzzy_labels

    # ─────────────────────────────────────────────────────────────────────────────
    # Compute inertia (WSS) given labels
    # ─────────────────────────────────────────────────────────────────────────────
    def inertia_from_labels(data, labels):
        s = 0.0
        for u in np.unique(labels):
            pts = data[labels == u]
            if pts.size == 0:
                continue
            center = pts.mean(axis=0)
            s += ((pts - center) ** 2).sum()
        return s

    # ─────────────────────────────────────────────────────────────────────────────
    # Pre-generate reference datasets for Gap statistic
    # ─────────────────────────────────────────────────────────────────────────────
    B = 5
    mins, maxs = z_mean.min(axis=0), z_mean.max(axis=0)
    refs = [np.random.uniform(mins, maxs, size=z_mean.shape) for _ in range(B)]

    def gap_stat(data, cluster_fn, k):
        # Compute Wk for actual data
        labels = cluster_fn(data, k)
        Wk = inertia_from_labels(data, labels)
        # Compute Wk for references
        Wk_refs = []
        for Xr in refs:
            try:
                lr = cluster_fn(Xr, k)
                Wk_refs.append(inertia_from_labels(Xr, lr))
            except:
                Wk_refs.append(np.nan)
        return np.log(np.nanmean(Wk_refs)) - np.log(Wk)

    # ─────────────────────────────────────────────────────────────────────────────
    # Sweep k and collect metrics
    # ─────────────────────────────────────────────────────────────────────────────
    ks = list(range(2, 21))
    results = {name: {"inertia": [], "ch": [], "gap": []} for name in methods}

    for name, fn in methods.items():
        print(f"Processing {name}...", end=" ")
        for k in ks:
            try:
                labels = fn(z_mean, k)
                results[name]["inertia"].append(inertia_from_labels(z_mean, labels))
                results[name]["ch"].append(calinski_harabasz_score(z_mean, labels))
                results[name]["gap"].append(gap_stat(z_mean, fn, k))
            except Exception as e:
                # keep array lengths consistent:
                results[name]["inertia"].append(np.nan)
                results[name]["ch"].append(np.nan)
                results[name]["gap"].append(np.nan)
        print("done")

# ─────────────────────────────────────────────────────────────────────────────
# Plot metrics
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

for name in methods:
    axes[0].plot(ks, results[name]["inertia"],    marker='o', label=name)
    axes[1].plot(ks, results[name]["ch"],         marker='o', label=name)
    axes[2].plot(ks, results[name]["gap"],        marker='o', label=name)

axes[0].set_ylabel("WSS (Inertia)")
axes[1].set_ylabel("Calinski–Harabasz")
axes[2].set_ylabel("Gap Statistic")
axes[2].set_xlabel("Number of Clusters (k)")

for ax in axes:
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

plt.tight_layout()
plt.savefig('/Users/dscqv/project/ssurgo_MO/SoilAnalysis/data/aggResult/clustering_matrix_0605.png', dpi=300, bbox_inches='tight')
#################################################################################################################################


    