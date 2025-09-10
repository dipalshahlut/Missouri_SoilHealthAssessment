#!/usr/bin/env python3
# spatial_maps.py
"""
Stage 4 — Spatial Visualization (robust path & join handling)

Inputs:
  - OUTPUT_DIR/main_with_best_labels_allAlgo.parquet
  - OUTPUT_DIR/best_k_for.json
  - A spatial layer (.gpkg or .shp) either passed via --spatial-path or auto-detected under --base-dir

Outputs:
  - OUTPUT_DIR/merged_clusters.gpkg
  - OUTPUT_DIR/figures/map_{method}_k{best_k}.png

 Usage:
python spatial_maps.py \
  --base-dir /path/to/data \
  --output-dir /path/to/data/aggResult
"""

import argparse
import json
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


# ---------- discovery helpers ----------

def _list_spatial_candidates(base_dir: Path) -> List[Path]:
    gps = list(base_dir.rglob("*.gpkg"))
    shps = list(base_dir.rglob("*.shp"))
    return gps + shps


def _pick_best_layer(path: Path, layer_name: Optional[str] = None) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gpkg":
        if layer_name:
            return gpd.read_file(path, layer=layer_name)
        try:
            # Try default (single-layer) read
            return gpd.read_file(path)
        except Exception:
            # Multi-layer: pick the layer with the most features
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


# ---------- inlined “merge_with_spatial_data” & “visualize_clusters_on_map” ----------

def merge_with_spatial_data(
    df_with_clusters: pd.DataFrame,
    base_dir: str,
    spatial_path: Optional[str] = None,
    spatial_layer: Optional[str] = None,
    df_key_forced: Optional[str] = None,
    gdf_key_forced: Optional[str] = None,
) -> gpd.GeoDataFrame:

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
        candidates = _list_spatial_candidates(base_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No spatial layer (.gpkg or .shp) found under {base_dir}.\n"
                "Tip: re-run with --spatial-path /full/path/to/your.gpkg"
            )
        # Log the first few candidates for visibility
        logging.info("Found %d spatial candidates under %s:", len(candidates), base_dir)
        for c in candidates[:10]:
            logging.info("  - %s", c)
        # Choose the one with the most features
# (inside merge_with_spatial_data, after listing candidates)

        # Score candidates: prefer files whose columns include MUKEY/mukey; break ties by feature count
        best_gdf, best_path, best_score = None, None, (-1, -1)  # (has_mukey, n_features)
        for c in candidates:
            try:
                tmp = _pick_best_layer(c, spatial_layer if c.suffix.lower()==".gpkg" else None)
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
    return merged


def visualize_clusters_on_map(
    gdf: gpd.GeoDataFrame,
    cluster_col: str,
    k: int,
    title: str,
    output_dir: str,
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


# ---------- main flow ----------

def run(
    base_dir: Path,
    output_dir: Path,
    methods: list[str],
    analysis_depth: int,
    spatial_path: Optional[Path],
    spatial_layer: Optional[str],
    df_key: Optional[str],
    gdf_key: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== Stage 4: Spatial Visualization ===")
    logging.info("Base dir       : %s", base_dir)
    logging.info("Output dir     : %s", output_dir)
    logging.info("Methods        : %s", methods)
    logging.info("Depth (cm)     : %d", analysis_depth)
    logging.info("Spatial path   : %s", spatial_path if spatial_path else "(auto-detect)")
    logging.info("Spatial layer  : %s", spatial_layer if spatial_layer else "(auto)")
    logging.info("Forced df key  : %s", df_key if df_key else "(auto)")
    logging.info("Forced gdf key : %s", gdf_key if gdf_key else "(auto)")

    # Stage-3 artifacts
    df_path = output_dir / "main_with_best_labels_allAlgo.parquet"
    best_k_path = output_dir / "best_k_for.json"
    if not df_path.exists():
        logging.error("Missing %s (run Stage 3 first).", df_path)
        sys.exit(1)
    if not best_k_path.exists():
        logging.error("Missing %s (run Stage 3 first).", best_k_path)
        sys.exit(1)

    df_with_clusters = pd.read_parquet(df_path)
    with open(best_k_path, "r") as f:
        best_k_for = json.load(f)

    logging.info("Loaded df_with_clusters: shape=%s", df_with_clusters.shape)
    logging.info("Loaded best_k_for: %s", best_k_for)

    # Merge
    merged_gdf = merge_with_spatial_data(
        df_with_clusters,
        str(base_dir),
        spatial_path=str(spatial_path) if spatial_path else None,
        spatial_layer=spatial_layer,
        df_key_forced=df_key,
        gdf_key_forced=gdf_key,
    )

    # Save merged layer
    gpkg_path = output_dir / "merged_clusters.gpkg"
    try:
        merged_gdf.to_file(gpkg_path, driver="GPKG")
        logging.info("Saved: %s (features=%d)", gpkg_path.name, len(merged_gdf))
    except Exception as e:
        logging.warning("Could not save GeoPackage: %s", e)

    # Render maps
    for method in methods:
        k = best_k_for.get(method)
        if k is None:
            logging.info("No best-k found for method '%s'; skipping map.", method)
            continue
        cluster_col = f"{method}_best{k}"
        if cluster_col not in merged_gdf.columns:
            logging.warning("Column '%s' not in merged data; skipping %s.", cluster_col, method)
            continue
        title = f"k={k} (VAE-{method}, {analysis_depth}cm)"
        try:
            out = visualize_clusters_on_map(
                merged_gdf, cluster_col=cluster_col, k=k, title=title, output_dir=str(output_dir)
            )
            logging.info("Saved map: %s", out.name)
        except Exception as e:
            logging.warning("Failed to render map for %s (k=%d): %s", method, k, e)

    logging.info("Stage 4 complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — Spatial Visualization (robust)")
    p.add_argument("--base-dir", type=Path, default=Path("/Users/dscqv/Desktop/SHA_copy/data"))
    p.add_argument("--output-dir", type=Path, default=Path("/Users/dscqv/Desktop/SHA_copy/data/aggResult"))
    p.add_argument("--methods", nargs="+", default=["KMeans", "Agglomerative", "Birch", "GMM"])
    p.add_argument("--analysis-depth", type=int, default=30)
    p.add_argument("--spatial-path", type=Path, help="Path to .gpkg or .shp; if missing, auto-detect under --base-dir")
    p.add_argument("--spatial-layer", type=str, help="Layer name inside a .gpkg (optional)")
    p.add_argument("--df-key", type=str, help="Force join key in the parquet dataframe (e.g., MUKEY)")
    p.add_argument("--gdf-key", type=str, help="Force join key in the spatial layer (e.g., MUKEY)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run(args.base_dir, args.output_dir, args.methods, args.analysis_depth, args.spatial_path, args.spatial_layer, args.df_key, args.gdf_key)
