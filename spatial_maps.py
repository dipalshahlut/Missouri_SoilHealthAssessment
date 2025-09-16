#!/usr/bin/env python3
# spatial_maps.py
"""
Stage 4 — Spatial Visualization (single method/k)

Inputs (from OUTPUT_DIR):
  - main_with_best_labels_allAlgo.parquet  (Stage 3)
  - z_mean.npy                              (Stage 2; only needed if we must compute labels)
  - A spatial layer (.gpkg or .shp) either passed via --spatial-path or auto-detected under --base-dir

Outputs (to OUTPUT_DIR):
  - shapefiles_with_data/MO_{analysis_depth}cm_clusters_vae_algorithms_merged_{method}_k{k}.csv
  - merged_clusters.gpkg
  - figures/map_{method}_best{k}.png

Usage:
python spatial_maps.py \
  --base-dir /path/to/data \
  --output-dir /path/to/data/aggResult \
  --analysis-depth 30 \
  --method KMeans \
  --k 10

With explicit spatial file ---->
python spatial_maps.py \
  --base-dir /path/to/data \
  --output-dir /path/to/data/aggResult \
  --analysis-depth 30 \
  --method Agglomerative \
  --k 12 \
  --spatial-path /path/to/data/mupoly.shp
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple, List

import pandas as pd

try:
    import geopandas as gpd
    import fiona
except Exception as e:
    raise SystemExit("geopandas & fiona are required: pip install geopandas fiona") from e

import matplotlib.pyplot as plt

# --- method canonicalization (so 'kmeans'/'KMEANS' etc. all work)
_CANON = {"kmeans": "KMeans", "agglomerative": "Agglomerative", "birch": "Birch", "gmm": "GMM"}
def _canon(m: str) -> str:
    return _CANON.get(m.strip().lower(), m)

# ---------- discovery helpers ----------

def _safe_unlink(p: Path) -> bool:
    """Delete file if present; return True if removed."""
    try:
        if p.exists():
            p.unlink()
            logging.info("Deleted existing file: %s", p)
            return True
    except Exception as e:
        logging.warning("Could not delete %s: %s", p, e)
    return False


def _list_spatial_candidates(base_dir: Path, exclude_dir: Optional[Path] = None) -> List[Path]:
    gps = list(base_dir.rglob("*.gpkg"))
    shps = list(base_dir.rglob("*.shp"))
    cands = gps + shps

    # 1) exclude anything under the pipeline output dir
    if exclude_dir:
        ex = Path(exclude_dir).resolve()
        cands = [p for p in cands if ex not in p.resolve().parents]

    # 2) drop obvious pipeline artifacts
    bad_names = {"merged_clusters.gpkg"}
    cands = [p for p in cands if p.name not in bad_names and "clusters_vae_algorithms_merged" not in p.name]
    return cands

def _pick_best_layer(path: Path, layer_name: Optional[str] = None) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gpkg":
        if layer_name:
            return gpd.read_file(path, layer=layer_name)
        try:
            return gpd.read_file(path)
        except Exception:
            # Multi-layer or unreadable default: choose layer with most features
            layers = fiona.listlayers(str(path))
            best, best_len = None, -1
            for lyr in layers:
                try:
                    tmp = gpd.read_file(path, layer=lyr)
                    if len(tmp) > best_len:
                        best, best_len = tmp, len(tmp)
                except Exception:
                    continue
            if best is None:
                raise
            return best
    # Shapefile or others supported by GeoPandas
    return gpd.read_file(path)


def _choose_join_key(df_cols, gdf_cols, forced_df_key: Optional[str], forced_gdf_key: Optional[str]) -> Optional[Tuple[str, str]]:
    if forced_df_key and forced_gdf_key:
        if forced_df_key in df_cols and forced_gdf_key in gdf_cols:
            return forced_df_key, forced_gdf_key
        else:
            logging.warning("Forced join keys not found (df: %s, gdf: %s). Falling back to heuristics.",
                            forced_df_key, forced_gdf_key)

    candidates = [
        ("MUKEY", "MUKEY"),
        ("mukey", "MUKEY"),
        ("mukey", "mukey"),
        ("unit_id", "unit_id"),
        ("poly_id", "poly_id"),
        ("GEOID", "GEOID"),
        ("OBJECTID", "OBJECTID"),
    ]
    df_lower = {c.lower(): c for c in df_cols}
    gdf_lower = {c.lower(): c for c in gdf_cols}
    for left, right in candidates:
        if left.lower() in df_lower and right.lower() in gdf_lower:
            return df_lower[left.lower()], gdf_lower[right.lower()]
    return None


# ---------- merge & map ----------

def merge_with_spatial_data(
    df_with_clusters: pd.DataFrame,
    base_dir: str | Path,
    spatial_path: Optional[str | Path] = None,
    spatial_layer: Optional[str] = None,
    df_key_forced: Optional[str] = None,
    gdf_key_forced: Optional[str] = None,
    save_csv: bool = True,
    output_dir: Optional[str | Path] = None,
    analysis_depth: Optional[int] = None,
    csv_method: Optional[str] = None,
    csv_k: Optional[int] = None,
    exclude_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:

    """
    Merge cluster labels to the best spatial layer found (or the one provided).

    If save_csv=True and output_dir is provided, writes a CSV under:
      <output_dir>/shapefiles_with_data/
        MO_{analysis_depth}cm_clusters_vae_algorithms_merged_{method}_k{k}.csv
    """
    base_dir = Path(base_dir)
    spatial_file: Optional[Path] = None

    if spatial_path:
        p = Path(spatial_path)
        if p.exists():
            spatial_file = p
            logging.info("Using explicit spatial file: %s", spatial_file)
        else:
            logging.warning("Spatial file not found at %s. Falling back to auto-detect.", p)

    if spatial_file is None:
        candidates = _list_spatial_candidates(base_dir, exclude_dir=exclude_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No spatial layer (.gpkg or .shp) found under {base_dir}.\n"
                "Tip: re-run with --spatial-path /full/path/to/your.gpkg"
            )
        logging.info("Found %d spatial candidates under %s:", len(candidates), base_dir)
        for c in candidates[:10]:
            logging.info("  - %s", c)

        # Score candidates: prefer MUKEY/mukey; break ties by feature count
        best_gdf, best_path, best_score = None, None, (-1, -1)  # (has_mukey, n_features)
        for c in candidates:
            try:
                tmp = _pick_best_layer(c, spatial_layer if c.suffix.lower() == ".gpkg" else None)
                cols_lower = {col.lower() for col in tmp.columns}
                has_mukey = 1 if ("mukey" in cols_lower) else 0
                n = len(tmp)
                score = (has_mukey, n)
                if score > best_score:
                    best_gdf, best_path, best_score = tmp, c, score
            except Exception:
                continue

        if best_gdf is None:
            raise FileNotFoundError("Could not open any discovered spatial files.")

        logging.info(
            "Auto-selected spatial file: %s (features=%d, has_mukey=%s)",
            best_path, len(best_gdf), "yes" if best_score[0] == 1 else "no"
        )
        gdf = best_gdf
    else:
        gdf = _pick_best_layer(spatial_file, spatial_layer)

    join = _choose_join_key(df_with_clusters.columns, gdf.columns, df_key_forced, gdf_key_forced)
    if not join:
        raise KeyError(
            "Could not find a common join key.\n"
            "Try forcing keys, e.g.: --df-key MUKEY --gdf-key MUKEY\n"
            f"df columns (sample): {list(df_with_clusters.columns)[:20]} ...\n"
            f"gdf columns (sample): {list(gdf.columns)[:20]} ..."
        )
    df_key, gdf_key = join
    logging.info("Joining on df['%s'] <-> gdf['%s']", df_key, gdf_key)

    # Align dtypes
    if df_with_clusters[df_key].dtype != gdf[gdf_key].dtype:
        gdf[gdf_key] = gdf[gdf_key].astype(str)
        df_with_clusters[df_key] = df_with_clusters[df_key].astype(str)

    merged = gdf.merge(df_with_clusters, how="left", left_on=gdf_key, right_on=df_key)

    # Normalize MUKEY/mukey to avoid GPKG duplicate-name error (keep spatial layer's MUKEY)
    if "MUKEY" in merged.columns and "mukey" in merged.columns:
        merged = merged.drop(columns=["mukey"])

    # --- optional CSV write with method/k-aware basename ---
    if save_csv:
        if not output_dir:
            logging.warning("save_csv=True but output_dir not provided; skipping CSV write.")
        else:
            outdir = Path(output_dir) / "shapefiles_with_data"
            outdir.mkdir(parents=True, exist_ok=True)

            depth_str = f"{int(analysis_depth)}cm" if analysis_depth is not None else "xxcm"
            method_part = str(csv_method) if csv_method else "Method"
            k_part = f"k{int(csv_k)}" if csv_k is not None else "kX"
            fname = f"MO_{depth_str}_clusters_vae_algorithms_merged_{method_part}_{k_part}.csv"

            csv_path = outdir / fname
            merged.drop(columns="geometry", errors="ignore").to_csv(csv_path, index=False)
            logging.info("Saved CSV: %s", csv_path)

    return merged


def visualize_clusters_on_map(
    gdf: gpd.GeoDataFrame,
    cluster_col: str,
    k: int,
    title: str,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = fig_dir / f"map_{cluster_col}.png"
    ax = gdf.plot(column=cluster_col, legend=True, figsize=(10, 8))
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------- main flow (single method/k) ----------

def run(
    base_dir: str | Path,
    output_dir: str | Path,
    analysis_depth: int,
    method: str,
    k: int,
    spatial_path: Optional[str | Path] = None,
    spatial_layer: Optional[str] = None,
    df_key: Optional[str] = None,
    gdf_key: Optional[str] = None,
) -> None:
    # Normalize to Path so .mkdir() etc. work even if strings were passed
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    # To ensure we don't reuse last run's merged output as an input
    _safe_unlink(output_dir / "merged_clusters.gpkg")
    method = _canon(method)
    k = int(k)

    logging.info("=== Stage 4: Spatial Visualization (single method/k) ===")
    logging.info("Base dir   : %s", base_dir)
    logging.info("Output dir : %s", output_dir)
    logging.info("Method/k   : %s, %d", method, k)

    # Stage-3 parquet
    df_path = output_dir / "main_with_best_labels_allAlgo.parquet"
    if not df_path.exists():
        logging.error("Missing %s (run Stage 3 first).", df_path)
        sys.exit(1)

    df = pd.read_parquet(df_path)

    # Ensure the requested label column exists; compute on-the-fly if absent
    col = f"{method}_best{k}"
    if col not in df.columns:
        z_path = output_dir / "z_mean.npy"
        if not z_path.exists():
            logging.error("Missing %s to compute labels. Run Stage 2 first.", z_path)
            sys.exit(1)
        import numpy as np
        from clustering_evaluation import fit_predict_labels
        z = np.load(z_path)
        if len(z) != len(df):
            logging.warning("z_mean length (%d) != df length (%d). Proceeding.", len(z), len(df))
        df[col] = fit_predict_labels(method, k, z, random_state=42)
        df.to_parquet(df_path, index=False)
        logging.info("Computed and added column: %s", col)

    # Merge & save ONE CSV with method/k suffix
    merged_gdf = merge_with_spatial_data(
        df,
        base_dir,
        spatial_path=spatial_path,
        spatial_layer=spatial_layer,
        df_key_forced=df_key,
        gdf_key_forced=gdf_key,
        save_csv=True,
        output_dir=output_dir,
        analysis_depth=analysis_depth,
        csv_method=method,
        csv_k=k,
        exclude_dir=output_dir, 
    )

    # Also save a merged GPKG snapshot for GIS use (ensure no dup col names)
    safe = merged_gdf.copy()
    # Drop any remaining case-insensitive duplicate column names (keep first)
    seen, drop_cols = set(), []
    for c in safe.columns:
        cl = c.lower()
        if cl in seen and c != "geometry":
            drop_cols.append(c)
        else:
            seen.add(cl)
    if drop_cols:
        safe = safe.drop(columns=drop_cols)
        logging.info("Dropped duplicate columns for GPKG: %s", drop_cols)

    gpkg_path = output_dir / "merged_clusters.gpkg"
    try:
        # Replace if exists (unlink for older GeoPandas/Fiona)
        if gpkg_path.exists():
            gpkg_path.unlink()
        safe.to_file(gpkg_path, driver="GPKG")
        logging.info("Saved: %s (features=%d)", gpkg_path.name, len(safe))
    except Exception as e:
        logging.warning("Could not save GeoPackage: %s", e)

    # Draw a single map
    title = f"k={k} (VAE-{method}, {analysis_depth}cm)"
    try:
        out = visualize_clusters_on_map(safe, cluster_col=col, k=k, title=title, output_dir=output_dir)
        logging.info("Saved map: %s", out.name)
    except Exception as e:
        logging.warning("Failed to render map: %s", e)

    logging.info("Stage 4 complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — Spatial Visualization (single method/k)")
    p.add_argument("--base-dir", type=Path, default=None, required=True)
    p.add_argument("--output-dir", type=Path, default=None, required=True)
    p.add_argument("--analysis-depth", type=int, default=30)
    p.add_argument("--method", type=str, required=True, help="Clustering method (e.g., KMeans, Agglomerative, Birch, GMM)")
    p.add_argument("--k", type=int, required=True, help="Number of clusters")
    p.add_argument("--spatial-path", type=Path, help="Path to .gpkg or .shp; if missing, auto-detect under --base-dir")
    p.add_argument("--spatial-layer", type=str, help="Layer name inside a .gpkg (optional)")
    p.add_argument("--df-key", type=str, help="Force join key in the parquet dataframe (e.g., MUKEY)")
    p.add_argument("--gdf-key", type=str, help="Force join key in the spatial layer (e.g., MUKEY)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        analysis_depth=args.analysis_depth,
        method=args.method,
        k=args.k,
        spatial_path=args.spatial_path,
        spatial_layer=args.spatial_layer,
        df_key=args.df_key,
        gdf_key=args.gdf_key,
    )
