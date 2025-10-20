#!/usr/bin/env python3
# pipeline_runbook.py
"""
************** End-to-end runbook ***************
SSURGO preprocessing → aggregation → VAE → Clustering → Visualization → Spatial Mapping
RUNBOOK & ORCHESTRATOR

Click Run (or `python pipeline_runbook.py`) to execute the pipeline.
Default behavior: run ALL steps (1–12) with your absolute paths.

Edit the CONFIG section below to change paths, steps, or defaults.
Supported methods (case-insensitive): "KMeans", "Agglomerative", "Birch", "GMM"

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations
import os
import sys, subprocess
import time
import traceback
from typing import Iterable
from pathlib import Path
import json, numpy as np, pandas as pd, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.ticker as mticker

# ───────────────────────────── CONFIG ─────────────────────────────
# Absolute paths for your machine (defaults from your earlier runs)
BASE_DIR = "/Users/Path/to/data"
OUTPUT_DIR = "/Users/Path/to/data/aggResult"

# Steps to run (1..12); default: all
# 1 aggregation.py
# 2 data_preparation.py
# 3 vae_training.py
# 4. clustering_evaluation.py
# 5 clustering_selection.py (optional scoring many ks/methods)
# 6 clustering_algorithms.py (one method, one k)
# 7. metrics_plots.py
# 8 latent_plots.py
# 9 visualization.py
# 10 spatial_maps.py (best labels mapping; optional)
# 11 spatial_mapping.py (one method,k mapping)
# 12. similarity_inxdex.py (compare two clustering method or k outputs)
# ===============================================================
# CONFIGURATION VARIABLES
# ===============================================================
# STEPS_TO_RUN: Controls which stages of the pipeline will run. 
#   Example: range(1, 13) runs all stages from 1 through 12.
#   You can also pass a subset like [1, 2, 5] to run only specific steps.
STEPS_TO_RUN: Iterable[int] = range(1, 13)
#STEPS_TO_RUN = [1,2,3,4,5,6,7,8,9,10]
# ANALYSIS_DEPTH: Soil analysis depth in centimeters.
#   Allowed values: 10, 30, or 100. Determines which depth slice of SSURGO data to use.
ANALYSIS_DEPTH = 30

# VAE_LATENT_DIM: Number of dimensions in the latent space of the VAE model.
#   A small value (e.g., 2) is good for visualization, larger values capture more complex structure.
VAE_LATENT_DIM = 2
# VAE_EPOCHS: Number of training epochs for the VAE.
#   More epochs may improve representation learning but will increase runtime.
VAE_EPOCHS = 100
# CLUSTER_METHOD: Clustering algorithm to use after VAE training.
#   Options: "KMeans", "Agglomerative", "Birch", "GMM".
#   Each method has different assumptions and behavior.
CLUSTER_METHOD = "KMeans" #  "KMeans", "Agglomerative", "Birch", "GMM"
# CLUSTER_K: Number of clusters for the chosen clustering method.S
#   Determines how many groups the data will be partitioned into.
CLUSTER_K = 10
# CONTINUE_ON_ERROR: If False, the pipeline will stop at the first error.
#   If True, it will attempt to continue running later steps even if one step fails.
#   recommended to keep it False otherwise it may use previous run intermediate step result
CONTINUE_ON_ERROR = False  # set True to keep going after errors
# EXCLUDE_MUKEYS: List of MUKEY (Map Unit Keys) to exclude from analysis.
#   Typically used to remove known problematic or irrelevant soil units.
EXCLUDE_MUKEYS = [2498901, 2498902, 2500799, 2500800, 2571513, 2571527]
# TARGET_CRS: Coordinate Reference System for geospatial outputs.
#   Default "EPSG:5070" is NAD83 / Conus Albers, suitable for US-wide soil datasets.
TARGET_CRS = "EPSG:5070"
# file_a, file_b: Paths to cluster assignment CSV files for similarity analysis.
#   These files should be outputs from Stage 10 (spatial_maps.py).
file_a="MO_30cm_clusters_vae_algorithms_merged_KMeans_k10.csv"
file_b="MO_30cm_clusters_vae_algorithms_merged_KMeans_k12.csv"
# col_a, col_b: Column names containing cluster labels in file_a and file_b.
#   Used to compute similarity indices (e.g., Adjusted Rand Index) between clusterings.
col_a="KMeans_best10"
col_b="KMeans_best12"
# ===============================================================
# ─────────────────────────── helpers ──────────────────────────────
def banner(msg: str):
    line = "─" * max(60, len(msg) + 9)
    print(f"\n{line}\n{msg}\n{line}")

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fail(msg: str, e: Exception):
    print(f" {msg}: {e}")
    traceback.print_exc()
    if not CONTINUE_ON_ERROR:
        sys.exit(1)

# ─────────────────────────── steps ────────────────────────────────
def step1_aggregation():
    banner("STEP 1 — aggregation.py (engineering & integration)")
    # Prefer imported aggregation function that accepts argv-style list
    try:
        from aggregation import main as run_main
        run_main(["--base-dir", BASE_DIR,
            "--output-dir", OUTPUT_DIR,
            "--target-crs", TARGET_CRS,
            "--analysis-depth", str(ANALYSIS_DEPTH),
        ])
        return run_main
    except Exception as e:
        fail("aggregation.py failed", e)

def step2_data_prep():
    banner("STEP 2 — data_preparation.py (Stage 1: prep/scaling)")
    try:
        # run as a separate process; no imports required
        cmd = [
            sys.executable, "data_preparation.py",
            "--base-dir", BASE_DIR,
            "--output-dir", OUTPUT_DIR,
            "--analysis-depth", str(ANALYSIS_DEPTH),
        ]
        if EXCLUDE_MUKEYS:
            cmd += ["--exclude-mukeys", *map(str, EXCLUDE_MUKEYS)]
            subprocess.run(cmd, check=True)
    except Exception as e:
        fail("data_preparation.py failed", e)


def step3_vae():
    banner("STEP 3 — vae_training.py (Stage 2: VAE & latents)")
    try:
        from vae_training import train_and_save
        train_and_save(Path(OUTPUT_DIR), VAE_LATENT_DIM, hidden_dim1=64,
        hidden_dim2=32, epochs=VAE_EPOCHS, batch_size=64, lr=1e-3)
    except Exception as e:
        fail("vae_training.py failed", e)

def step4_cluster_eval():
    banner("STEP 4 — clustering_evaluation.py (evaluate methods & best k)")
    try:
        from clustering_evaluation import evaluate_multiple_algorithms, get_best_clustering_results
        z = np.load(os.path.join(OUTPUT_DIR, "z_mean.npy"))
        base_df = (pd.read_parquet(os.path.join(OUTPUT_DIR, "prepared_df.parquet"))
                   if os.path.exists(os.path.join(OUTPUT_DIR, "prepared_df.parquet"))
                   else pd.read_csv(os.path.join(OUTPUT_DIR, "main_df.csv")))
        methods = ["KMeans", "Agglomerative", "Birch", "GMM"]

        scores, best_k_for, best_score_for = evaluate_multiple_algorithms(
            z, methods=methods, k_range=range(2, 21), random_state=42)
        # Save artifacts
        recs = [{"method": m, "metric": met, "k": k, "value": v}
                for m, mm in scores.items()
                for met, kv in mm.items()
                for k, v in kv.items()]
        pd.DataFrame(recs).to_csv(os.path.join(OUTPUT_DIR, "clustering_scores.csv"), index=False)
        json.dump(best_k_for, open(os.path.join(OUTPUT_DIR, "best_k_for.json"), "w"), indent=2)
        json.dump(best_score_for, open(os.path.join(OUTPUT_DIR, "best_score_for.json"), "w"), indent=2)

        df_with = get_best_clustering_results(base_df, z, best_k_for, methods, random_state=42)
        df_with.to_parquet(os.path.join(OUTPUT_DIR, "main_with_best_labels_allAlgo.parquet"), index=False)
        print("✅ clustering_evaluation done.")
    except Exception as e:
        fail("clustering_evaluation.py failed", e)

def step5_cluster_select():
    banner("STEP 5 — clustering_selection.py (optional: score many ks/methods)")
    try:
        try:
            from clustering_selection import score_many_clusterings
            from clustering_selection import main as run_select
            return score_many_clusterings(
                output_dir=OUTPUT_DIR,
                methods=["KMeans", "Agglomerative", "Birch", "GMM"],
                k_min=2, k_max=20
            )
        except ImportError:
            from clustering_selection import main as run_select
            argv = [
                "--output-dir", OUTPUT_DIR,
                "--methods", "KMeans", "Agglomerative", "Birch", "GMM",
                "--k-min", "2", "--k-max", "20",
            ]
            return run_select(argv)
    except Exception as e:
        fail("clustering_selection.py failed", e)

def step6_cluster_single():
    banner("STEP 6 — clustering_algorithms.py (single method & k)")
    try:
        from clustering_algorithms import run_single_clustering_io
        labels, sil, col = run_single_clustering_io(
            output_dir=OUTPUT_DIR,
            method=CLUSTER_METHOD,
            k=CLUSTER_K,
        )
        print(f"✅ Clustering done. Method={CLUSTER_METHOD}, k={CLUSTER_K}, silhouette={sil}")
        return labels, sil, col
    except Exception as e:
        fail("clustering_algorithms.py failed", e)

def step7_metrics_plots():
    banner("STEP 7 — metric_plots.py (plot evaluation metrics)")
    import metric_plots
    return metric_plots.run(base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        scaled_path=os.path.join(OUTPUT_DIR, "z_mean.npy"),
        methods=["KMeans", "Agglomerative", "Birch", "GMM"],
        k_min=2, k_max=20, gap_B=5, verbosity=2
    )

def step8_latent_plots():
    banner("STEP 8 — latent_plots.py (2D latent plots)")
    try:
        try:
            from latent_plots import plot_latent
            return plot_latent(
                output_dir=OUTPUT_DIR,
                method=CLUSTER_METHOD,  # or None to auto-pick best
                k=CLUSTER_K,            # or None to use best_k_for.json
                title=f"{CLUSTER_METHOD} (k={CLUSTER_K}) clustering plot"
            )
        except ImportError:
            from latent_plots import main as run_latent
            argv = [
                "--output-dir", OUTPUT_DIR,
                "--method", CLUSTER_METHOD,
                "--k", str(CLUSTER_K),
            ]
            return run_latent(argv)
    except Exception as e:
        fail("latent_plots.py failed", e)

def step9_visualization():
    banner("STEP 9 — visualization.py (scatter, boxplots, area-by-cluster)")
    try:
        from Visualization import make_all_plots
        return make_all_plots(output_dir=OUTPUT_DIR, method=CLUSTER_METHOD, k=CLUSTER_K, exclude_mukeys=EXCLUDE_MUKEYS)
    except Exception as e:
        fail("visualization.py failed", e)

def step10_spatial_maps():
    banner("STEP 10 — spatial_maps.py (best labels mapping)")
    try:
        try:
            from spatial_maps import run as spatial_run
            return spatial_run(base_dir=Path(BASE_DIR), output_dir=Path(OUTPUT_DIR),analysis_depth=ANALYSIS_DEPTH, method=CLUSTER_METHOD, 
                               k=int(CLUSTER_K), spatial_path=Path(BASE_DIR)/"mupoly.shp", spatial_layer=None, df_key="mukey", gdf_key="MUKEY")
        except ImportError:
            # optional; not all setups include this stage
            print("ℹ️ spatial_maps.py not available or helper missing — skipping.")
    except Exception as e:
        fail("spatial_maps.py failed", e)

def step11_spatial_mapping():
    banner("STEP 11 — spatial_mapping.py (one method, k mapping)")
    try:
        import spatial_mapping as sm
        # If your helper supports (base_dir, output_dir, method, k)
        return sm.create_spatial_products(
            output_dir=OUTPUT_DIR,
            method=CLUSTER_METHOD,
            k=CLUSTER_K,
            vector_path=f"{BASE_DIR}/mupoly.shp",  # or .gpkg
            target_crs="EPSG:5070",
            write_gpkg=True,
            write_shp=True,
            shp_minimal=True,
        )
    except Exception as e:
        fail("spatial_mapping.py failed", e)

def step12_similarity_index():
    banner("STEP 12 — similarity_index.py (compare two clustering method or k outputs)")
    try:
        from similarity_index import run_similarity_index
        input_dir = Path(OUTPUT_DIR) / "shapefiles_with_data"
        pattern = "*clusters_vae_algorithms_merged_*_k*.csv"
        csvs = sorted(input_dir.glob(pattern))

        if len(csvs) < 2:
            msg = (
                f"   Similarity Index step skipped.\n"
                f"   ➤ This step runs *only when there are at least TWO clustering results* to compare.\n"
                f"   ➤ Currently found {len(csvs)} file(s) in:\n"
                f"       {input_dir}\n"
                f"   ➤ To use this step, first run your clustering step with a *different k* or a *different algorithm*.\n"
                f"   ➤ This will generate multiple clustering result files matching the pattern:\n"
                f"       {pattern}\n")
            print(msg)
            print("Similarity Index skipped: found %d clustering file(s) in %s (need ≥ 2).",
                len(csvs), input_dir)
            return

        # Pick two most recent automatically
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        file_a = csvs[0].name
        file_b = csvs[1].name

        res = run_similarity_index(
            input_dir=input_dir,
            file_a=file_a,
            file_b=file_b,
        )
        print(f"✅ ARI similarity computed: {res['ari']:.4f}")
        print(f"Counts CSV: {res['counts_csv']}")
        print(f"Row% CSV : {res['rowpct_csv']}")
        print(f"HTML     : {res['html']}")
        print(f"PNG      : {res['png']}")

    except Exception as e:
        fail("Similarity Index failed", e)


# ─────────────────────────── driver ───────────────────────────────
def run():
    t0 = time.time()
    print(f"BASE_DIR   = {BASE_DIR}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}  (will be created if missing)")

    # Ensure output/working directory
    ensure_output_dir(OUTPUT_DIR)

    step_funcs = {
        1: step1_aggregation,
        2: step2_data_prep,
        3: step3_vae,
        4: step4_cluster_eval,
        5: step5_cluster_select,
        6: step6_cluster_single,
        7: step7_metrics_plots,
        8: step8_latent_plots,
        9: step9_visualization,
        10: step10_spatial_maps,
        11: step11_spatial_mapping,
        12: step12_similarity_index,
    }

    for s in STEPS_TO_RUN:
        func = step_funcs.get(s)
        if not func:
            print(f"⚠️ Unknown step {s}, skipping.")
            continue
        _t = time.time()
        func()
        print(f"⏱️  Step {s} finished in {time.time()-_t:.1f}s")

    print(f"\n✅ Pipeline completed in {time.time()-t0:.1f}s")
    print("Result artifacts are in:", OUTPUT_DIR)

if __name__ == "__main__":
    run()
