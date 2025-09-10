#!/usr/bin/env python3
"""
spatial_mapping.py

Merge clustering labels onto Missouri MU polygon geometries and export spatial products.

Inputs
------
- OUTPUT_DIR/main_df_with_{method}_k{k}.csv  # produced by clustering_algorithms.py
- --vector-path <path to MU polygons>        # e.g., mupoly.shp or .gpkg
  (Supports: Shapefile, GeoPackage, and other formats readable by GeoPandas)

Outputs (written to OUTPUT_DIR)
-------------------------------
- shapefiles_with_data/MO_30cm_clusters_{method}_k{k}.gpkg
- shapefiles_with_data/shp/MO_30cm_clusters_{method}_k{k}.shp   
- map_{method}_k{k}.png

Notes
-----
- Join key is 'mukey'. If your vector file has 'MUKEY', it will be renamed to 'mukey'.
- Reprojects to EPSG:5070 by default to ensure equal-area mapping and area computation.
- ESRI Shapefile has 10-character field name limits. We duplicate the cluster labels
  into a safe field named 'cluster' in the Shapefile to avoid truncation issues.

Usage:
python spatial_mapping.py \
  --output-dir /path/to/data/aggResult \
  --method KMeans \
  --k 10 \
  --vector-path /path/to/data/mupoly.shp
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from typing import Optional

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# --------------------------- logging ---------------------------

def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.INFO
    if verbosity >= 3:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------- io helpers ---------------------------

def _df_with_labels_path(output_dir: str, method: str, k: int) -> str:
    return os.path.join(output_dir, f"main_df_with_{method}_k{k}.csv")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# --------------------------- spatial helpers ---------------------------

def _load_vector(vector_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"Vector file loaded but empty: {vector_path}")
    if gdf.geometry.is_empty.any():
        logging.warning("Some geometries are empty in the input vector.")
    # Normalize join key
    cols = {c.lower(): c for c in gdf.columns}
    if "mukey" in cols:
        key_col = cols["mukey"]
        if key_col != "mukey":
            gdf = gdf.rename(columns={key_col: "mukey"})
    elif "mukey" not in gdf.columns and "MUKEY" in gdf.columns:
        gdf = gdf.rename(columns={"MUKEY": "mukey"})
    elif "mukey" not in gdf.columns:
        guess = [c for c in gdf.columns if c.lower() == "mukey"]
        if guess:
            gdf = gdf.rename(columns={guess[0]: "mukey"})
        else:
            raise KeyError(
                "No 'mukey' (or 'MUKEY') column found in the vector data. "
                "Please ensure your polygons contain a MU key column named 'mukey' or 'MUKEY'."
            )
    gdf["mukey"] = gdf["mukey"].astype(str)
    return gdf

def _reproject_and_area(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    if gdf.crs is None:
        logging.warning("Input vector has no CRS; assuming EPSG:4326 (WGS84).")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    # Compute area in CRS units (EPSG:5070 → square meters)
    if "area_m2" not in gdf.columns:
        gdf["area_m2"] = gdf.geometry.area
    # Provide acres for convenience
    if "area_ac" not in gdf.columns:
        gdf["area_ac"] = gdf["area_m2"] * 0.000247105381
    return gdf


# --------------------------- core ---------------------------

def create_spatial_products(
    output_dir: str,
    method: str,
    k: int,
    vector_path: str,
    target_crs: str = "EPSG:5070",
    map_cmap: str = "viridis",
    *,
    write_gpkg: bool = True,
    write_shp: bool = True,
    shp_minimal: bool = True
) -> dict:
    """
    Merge clustered labels onto polygons and write a GPKG plus a static PNG map.
    Optionally also write an ESRI Shapefile (.shp).

    Parameters
    ----------
    write_gpkg : bool
        Write GeoPackage output (default True).
    write_shp : bool
        Write an ESRI Shapefile (.shp) (default True).
    shp_minimal : bool
        If True, the Shapefile will contain only a minimal set of safe fields:
        ['mukey', 'cluster'] + geometry. This avoids 10-char field name limits.
        If False, will attempt to write all fields (may be truncated by driver).

    Returns
    -------
    dict with keys: gpkg_path, shp_path, png_path, merged_count
    """
    labels_col = f"{method}_k{k}"
    df_path = _df_with_labels_path(output_dir, method, k)

    # Load tabular results
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing clustered dataframe: {df_path}")
    df = pd.read_csv(df_path)
    if labels_col not in df.columns:
        raise KeyError(f"Column '{labels_col}' is missing in {df_path}")
    if "mukey" not in df.columns:
        raise KeyError("main_df_with_* is missing 'mukey' column required for spatial join.")
    df["mukey"] = df["mukey"].astype(str)
    logging.info("Loaded DF with labels: %s (shape=%s)", os.path.basename(df_path), df.shape)

    # Load vectors
    gdf = _load_vector(vector_path)
    gdf = _reproject_and_area(gdf, target_crs=target_crs)

    # Join
    gdfm = gdf.merge(df[["mukey", labels_col]], on="mukey", how="left")
    merged_count = int(gdfm[labels_col].notna().sum())
    if merged_count == 0:
        logging.warning(
            "No polygons received a label (check MU keys). "
            "mukey examples (vector): %s ; (df): %s",
            gdf["mukey"].head(3).tolist(),
            df["mukey"].head(3).tolist()
        )

    shp_base_dir = os.path.join(output_dir, "shapefiles_with_data")
    _ensure_dir(shp_base_dir)
    base = f"MO_30cm_clusters_{method}_k{k}"

    gpkg_path = None
    shp_path = None

    # Write GeoPackage
    if write_gpkg:
        gpkg_path = os.path.join(shp_base_dir, f"{base}.gpkg")
        # If file exists, try to remove to avoid layer conflicts
        if os.path.exists(gpkg_path):
            try:
                os.remove(gpkg_path)
            except Exception as e:
                logging.warning("Could not remove existing GPKG, will overwrite layers: %s", e)
        gdfm.to_file(gpkg_path, layer=base, driver="GPKG")
        logging.info("Wrote: %s", gpkg_path)

    # Write ESRI Shapefile (with safe fields)
    if write_shp:
        shp_dir = os.path.join(shp_base_dir, "shp")
        _ensure_dir(shp_dir)
        shp_path = os.path.join(shp_dir, f"{base}.shp")

        # Build a shapefile-safe GeoDataFrame:
        #  - duplicate cluster into a <=10 char col name 'cluster'
        #  - include mukey (<=10)
        #  - optionally keep minimal fields to avoid truncation issues
        gdf_shp = gdfm.copy()
        gdf_shp["cluster"] = gdf_shp[labels_col]
        keep_cols = ["mukey", "cluster", "geometry"] if shp_minimal else list(gdf_shp.columns)
        gdf_shp = gdf_shp[keep_cols]

        # Write Shapefile
        # Note: driver will truncate long field names automatically, but we proactively keep them safe.
        if os.path.exists(shp_path):
            # remove prior shapefile sidecars if present
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = shp_path.replace(".shp", ext)
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass
        gdf_shp.to_file(shp_path, driver="ESRI Shapefile")
        logging.info("Wrote: %s", shp_path)

    # Quick PNG map
    png_path = os.path.join(output_dir, f"map_{method}_k{k}.png")
    try:
        ax = gdfm.plot(column=labels_col, legend=True, cmap=map_cmap,
                       figsize=(10, 8), alpha=0.9, linewidth=0)
        ax.set_axis_off()
        plt.title(f"Clusters — {method} (k={k})")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        logging.info("Wrote: %s", png_path)
    except Exception as e:
        logging.warning("PNG map generation failed: %s", e)
        png_path = None

    return {"gpkg_path": gpkg_path, "shp_path": shp_path, "png_path": png_path, "merged_count": merged_count}

# ---------------------------------------------------------------------
# Convenience wrapper for programmatic use
# ---------------------------------------------------------------------
def make_spatial_products(base_dir: str, output_dir: str, method: str, k: int, **kwargs):
    """
    Convenience wrapper around create_spatial_products to allow:
        make_spatial_products(BASE_DIR, OUTPUT_DIR, method="KMeans", k=10)

    Parameters
    ----------
    base_dir : str
        Path to input data (where mupoly.shp or mupoly.gpkg is located).
    output_dir : str
        Directory where cluster results and outputs are written.
    method : str
        Clustering method name (e.g., "KMeans", "Agglomerative").
    k : int
        Number of clusters.
    kwargs : dict
        Extra arguments passed through to create_spatial_products
        (e.g., target_crs="EPSG:5070", write_shp=True/False, shp_minimal=True/False).

    Notes
    -----
    - This helper constructs the vector_path automatically as
      f"{base_dir}/mupoly.shp".
    - If your polygons are stored in .gpkg, you can override:
        make_spatial_products(..., vector_path=f"{base_dir}/mupoly.gpkg")
    """
    vector_path = kwargs.pop("vector_path", f"{base_dir}/mupoly.shp")
    return create_spatial_products(
        output_dir=output_dir,
        method=method,
        k=k,
        vector_path=vector_path,
        **kwargs
    )


# --------------------------- CLI ---------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge clustering labels with MU polygons and export spatial products."
    )
    p.add_argument("-o", "--output-dir", required=True,
                   help="Directory containing main_df_with_{method}_k{k}.csv")
    p.add_argument("-m", "--method", required=True,
                   choices=["KMeans", "Agglomerative", "Birch", "GMM", "FuzzyCMeans"],
                   help="Clustering method used.")
    p.add_argument("-k", "--k", type=int, required=True,
                   help="k used in clustering.")
    p.add_argument("--vector-path", required=True,
                   help="Path to MU polygon vector file (e.g., mupoly.shp or .gpkg).")
    p.add_argument("--target-crs", default="EPSG:5070",
                   help="Target CRS for reprojection and area calc (default: EPSG:5070).")
    p.add_argument("--no-gpkg", action="store_true", help="Do not write GeoPackage output.")
    p.add_argument("--no-shp", action="store_true", help="Do not write ESRI Shapefile output.")
    p.add_argument("--shp-full", action="store_true",
                   help="Write Shapefile with all fields (may truncate names). Default is minimal fields.")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase verbosity (-v, -vv).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    _setup_logging(args.verbose)

    try:
        out = create_spatial_products(
            output_dir=args.output_dir,
            method=args.method,
            k=args.k,
            vector_path=args.vector_path,
            target_crs=args.target_crs,
            write_gpkg=(not args.no_gpkg),
            write_shp=(not args.no_shp),
            shp_minimal=(not args.shp_full),
        )
        print("✅ Spatial mapping completed successfully.")
        print("Results saved in:", args.output_dir)
        if out.get("gpkg_path"):
            print(f"  - GeoPackage: {out['gpkg_path']}")
        if out.get("shp_path"):
            print(f"  - Shapefile : {out['shp_path']}")
        if out.get("png_path"):
            print(f"  - Map PNG   : {out['png_path']}")
        print(f"  - Polygons with labels: {out['merged_count']}")
        return 0
    except Exception as e:
        logging.exception("Spatial mapping failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
