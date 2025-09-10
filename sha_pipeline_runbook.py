#!/usr/bin/env python3
# pipeline_runbook.py
"""
Soil VAE → Clustering → Visualization → Spatial Mapping
RUNBOOK & ORCHESTRATOR

👉 Click Run (or `python pipeline_runbook.py`) to execute the pipeline.
Default behavior: run ALL steps (1–9) with your absolute paths.

Edit the CONFIG section below to change paths, steps, or defaults.
"""

from __future__ import annotations
import os
import sys, subprocess
import time
import traceback
from typing import Iterable
from pathlib import Path
from vae_training import train_and_save
from clustering_selection import score_many_clusterings
from clustering_selection import main as run_select
from clustering_algorithms import run_single_clustering_io
from Visualization import make_all_plots
# ───────────────────────────── CONFIG ─────────────────────────────
# Absolute paths for your machine (defaults from your earlier runs)
BASE_DIR = "/Users/you/Absolute/Path/to/data"
OUTPUT_DIR = "/Users/you/Absolute/Path/to/data/aggResult"

# Steps to run (1..9); default: all
# 1 main.py
# 2 data_preparation.py
# 3 vae_training.py
# 4 clustering_selection.py (optional scoring many ks/methods)
# 5 clustering_algorithms.py (one method, one k)
# 6 latent_plots.py
# 7 visualization.py
# 8 spatial_maps.py (best labels mapping; optional)
# 9 spatial_mapping.py (one method,k mapping)
STEPS_TO_RUN: Iterable[int] = range(1, 10)

# Defaults you can tweak
ANALYSIS_DEPTH = 30
VAE_LATENT_DIM = 2
VAE_EPOCHS = 100
CLUSTER_METHOD = "KMeans"
CLUSTER_K = 10
CONTINUE_ON_ERROR = False  # set True to keep going after errors

# ─────────────────────────── helpers ──────────────────────────────
def banner(msg: str):
    line = "─" * max(60, len(msg) + 8)
    print(f"\n{line}\n{msg}\n{line}")

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fail(msg: str, e: Exception):
    print(f"❌ {msg}: {e}")
    traceback.print_exc()
    if not CONTINUE_ON_ERROR:
        sys.exit(1)

# ─────────────────────────── steps ────────────────────────────────
def step1_main():
    banner("STEP 1 — main.py (engineering & integration)")
    # Prefer imported main function that accepts argv-style list
    try:
        from main import main as run_main
        argv = [
            "--base-dir", BASE_DIR,
            "--output-dir", OUTPUT_DIR,
            "--analysis-depth", str(ANALYSIS_DEPTH),
            "-vv",
        ]
        return run_main(argv)
    except Exception as e:
        fail("main.py failed", e)

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
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    except Exception as e:
        fail("data_preparation.py failed", e)


def step3_vae():
    banner("STEP 3 — vae_training.py (Stage 2: VAE & latents)")
    try:
        train_and_save(Path(OUTPUT_DIR), VAE_LATENT_DIM, VAE_EPOCHS)
    except Exception as e:
        fail("vae_training.py failed", e)

def step4_cluster_select():
    banner("STEP 4 — clustering_selection.py (optional: score many ks/methods)")
    try:
        try:
            from clustering_selection import score_many_clusterings
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

def step5_cluster_single():
    banner("STEP 5 — clustering_algorithms.py (single method & k)")
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

def step6_latent_plots():
    banner("STEP 6 — latent_plots.py (2D latent plots)")
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

def step7_visualization():
    banner("STEP 7 — visualization.py (scatter, boxplots, area-by-cluster)")
    try:
        
        return make_all_plots(output_dir=OUTPUT_DIR, method=CLUSTER_METHOD, k=CLUSTER_K)
    except Exception as e:
        fail("visualization.py failed", e)

def step8_spatial_maps():
    banner("STEP 8 — spatial_maps.py (best labels mapping)")
    try:
        try:
            from spatial_maps import build_and_map_best
            return build_and_map_best(base_dir=BASE_DIR, output_dir=OUTPUT_DIR)
        except ImportError:
            # optional; not all setups include this stage
            print("ℹ️ spatial_maps.py not available or helper missing — skipping.")
    except Exception as e:
        fail("spatial_maps.py failed", e)

def step9_spatial_mapping():
    banner("STEP 9 — spatial_mapping.py (one method,k mapping)")
    try:
        try:
            from spatial_mapping import make_spatial_products
        except ImportError:
            # as discussed earlier, your module may expose `create_spatial_products`
            from spatial_mapping import create_spatial_products as make_spatial_products

        # If your helper supports (base_dir, output_dir, method, k)
        return make_spatial_products(
            base_dir=BASE_DIR,
            output_dir=OUTPUT_DIR,
            method="KMeans",
            k=10,
            write_shp=True,      # default True now
            shp_minimal=True     # keep only safe minimal fields in .shp
        )
    except Exception as e:
        fail("spatial_mapping.py failed", e)

# ─────────────────────────── driver ───────────────────────────────
def run():
    t0 = time.time()
    print(f"BASE_DIR   = {BASE_DIR}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}  (will be created if missing)")

    # Ensure output/working directory
    ensure_output_dir(OUTPUT_DIR)

    step_funcs = {
        1: step1_main,
        2: step2_data_prep,
        3: step3_vae,
        4: step4_cluster_select,
        5: step5_cluster_single,
        6: step6_latent_plots,
        7: step7_visualization,
        8: step8_spatial_maps,
        9: step9_spatial_mapping,
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
    print("Artifacts are in:", OUTPUT_DIR)

if __name__ == "__main__":
    run()
