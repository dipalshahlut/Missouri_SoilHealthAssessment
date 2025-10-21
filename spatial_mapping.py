#!/usr/bin/env python3
"""
spatial_mapping.py

Merge clustering labels onto Missouri MU polygon geometries and export spatial products.

Inputs
------
EITHER:
  - (Recommended) Pass the exact file via --labels-csv and column via --labels-col
OR
  - Place a CSV in OUTPUT_DIR (or subfolders) that matches patterns like:
      main_df_with_{method}_k{k}.csv
      main_df_with_{method}_best{k}.csv
      *{method}*k{k}*.csv
    The script will auto-discover it.

CSV requirements:
  - Must contain 'mukey' and a labels column (e.g., 'KMeans_k10' or 'KMeans_best10')

Vector:
  - --vector-path <path to MU polygons> (Shapefile .shp or GeoPackage .gpkg)

Outputs (written to OUTPUT_DIR)
-------------------------------
- shapefile_with_data/MO_30cm_clusters_{method}_k{k}.gpkg
- shapefile_with_data/shp/MO_30cm_clusters_{method}_k{k}.shp
- map_{method}_k{k}.png

Notes
-----
- Join key is 'mukey'. If your vector file has 'MUKEY', it will be renamed to 'mukey'.
- Reprojects to EPSG:5070 by default to ensure equal-area mapping and area computation.
- ESRI Shapefile has 10-character field name limits; we also write 'cluster' as a safe field.

Usage:
  python spatial_mapping.py \
    --output-dir /path/to/data/aggResult \
    --method KMeans \
    --k 10 \
    --vector-path /path/to/data/mupoly.shp \
    --out-basename mupoly_10 \
    --shp-folder-name shapefile_with_data
"""

from __future__ import annotations

import os
import re
import sys
import glob
import argparse
import logging
from typing import Optional
from pathlib import Path

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

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _guess_driver_from_path(p: str) -> str:
    ext = Path(p).suffix.lower()
    if ext == ".gpkg":
        return "GPKG"
    if ext == ".shp":
        return "ESRI Shapefile"
    # default to GPKG if unknown but readable by GeoPandas
    return "GPKG"


# --------------------------- discovery helpers ---------------------------

def _find_labels_csv(output_dir: str, method: str, k: int, labels_csv: Optional[str]) -> str:
    """Return a valid labels CSV path. Prefer explicit path; else discover."""
    if labels_csv:
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"--labels-csv not found: {labels_csv}")
        return labels_csv

    # Try a few likely exact names in the root of output_dir
    candidates = [
        os.path.join(output_dir, f"main_df_with_{method}_k{k}.csv"),
        os.path.join(output_dir, f"main_df_with_{method}_best{k}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            logging.info("Using labels CSV: %s", c)
            return c

    # Recursive glob search inside output_dir for anything that looks right
    glob_patterns = [
        f"**/*{method}*k{k}*.csv",
        f"**/main_df_with_*{method}*k{k}*.csv",
        f"**/MO_30cm_clusters_*{method}*k{k}*.csv",
    ]
    found: list[str] = []
    for pat in glob_patterns:
        found.extend(glob.glob(os.path.join(output_dir, pat), recursive=True))

    # Keep the ones that actually have 'mukey' in header (quick sniff)
    viable = []
    for p in found:
        try:
            head = pd.read_csv(p, nrows=5)
            if any(c.lower() == "mukey" for c in head.columns):
                viable.append(p)
        except Exception:
            continue

    if not viable:
        raise FileNotFoundError(
            "Could not locate a clustered dataframe. Looked for patterns like:\n"
            f"  - {output_dir}/main_df_with_{method}_k{k}.csv\n"
            f"  - {output_dir}/main_df_with_{method}_best{k}.csv\n"
            f"  - {output_dir}/**/*{method}*k{k}*.csv\n"
            "Tip: pass --labels-csv explicitly."
        )

    # Prefer shortest path / nearest match
    viable.sort(key=lambda p: (p.count(os.sep), len(Path(p).name)))
    logging.info("Auto-discovered labels CSV: %s", viable[0])
    return viable[0]


_LABEL_COL_PATTERNS = [
    # exact first
    lambda method, k: f"{method}_k{k}",
    lambda method, k: f"{method}_best{k}",
    # common alternates
    lambda method, k: f"{method}_best_{k}",
    lambda method, k: f"{method}{k}",
    lambda method, k: f"{method}_k_{k}",
]


def _guess_labels_col(df: pd.DataFrame, method: str, k: int, labels_col: Optional[str]) -> str:
    """Return the label column name, trying explicit -> patterns -> regex -> heuristic."""
    if labels_col:
        if labels_col not in df.columns:
            raise KeyError(f"--labels-col '{labels_col}' not found in CSV columns: {list(df.columns)}")
        return labels_col

    # 1) pattern hits
    for maker in _LABEL_COL_PATTERNS:
        name = maker(method, k)
        if name in df.columns:
            return name

    # 2) common generic field
    if "cluster" in df.columns:
        return "cluster"

    # 3) regex search like r"(?i)^KMeans.*(?:_)?k(?:_)?10$"
    rx = re.compile(rf"(?i)^{re.escape(method)}.*k_?{k}$")
    for c in df.columns:
        if rx.search(str(c)):
            return c

    # 4) heuristic: numeric-ish column with exactly k unique labels
    #    (tolerate NaNs; ignore id-like columns)
    plausible = []
    for c in df.columns:
        if c.lower() in {"mukey", "mu_key", "objectid", "fid"}:
            continue
        s = df[c]
        # try to cast to int codes if necessary
        try:
            nunique = s.dropna().nunique()
            # clusters typically have k unique values (0..k-1 or 1..k); allow k or k+1 (rare)
            if 2 <= nunique <= max(k, 2) and abs(nunique - k) <= 1:
                plausible.append((c, nunique))
        except Exception:
            continue

    if plausible:
        # prefer the one with nunique closest to k, then shorter name
        plausible.sort(key=lambda t: (abs(t[1] - k), len(t[0])))
        logging.warning("Heuristically selected labels column: %s (unique=%d)", plausible[0][0], plausible[0][1])
        return plausible[0][0]

    raise KeyError(
        "Could not guess the labels column. Try passing --labels-col explicitly. "
        f"CSV columns: {list(df.columns)}"
    )


# --------------------------- spatial helpers ---------------------------

def _load_vector(vector_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"Vector file loaded but empty: {vector_path}")
    if gdf.geometry.is_empty.any():
        logging.warning("Some geometries are empty in the input vector.")

    # Normalize join key (-> 'mukey' as string)
    cols_lower = {c.lower(): c for c in gdf.columns}
    if "mukey" in cols_lower:
        key_col = cols_lower["mukey"]
        if key_col != "mukey":
            gdf = gdf.rename(columns={key_col: "mukey"})
    elif "MUKEY" in gdf.columns:
        gdf = gdf.rename(columns={"MUKEY": "mukey"})
    else:
        # last-ditch case-insensitive search
        guess = [c for c in gdf.columns if c.lower() == "mukey"]
        if guess:
            gdf = gdf.rename(columns={guess[0]: "mukey"})
        else:
            raise KeyError("No 'mukey' (or 'MUKEY') column found in the vector data.")

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
    if "area_ac" not in gdf.columns:
        gdf["area_ac"] = gdf["area_m2"] * 0.000247105381

    gdf["area_m2"] = gdf["area_m2"].round(0).astype("int64")   #
    gdf["area_ac"] = gdf["area_ac"].round(2)      
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
    labels_csv: Optional[str] = None,
    labels_col: Optional[str] = None,
    write_gpkg: bool = True,
    write_shp: bool = True,
    shp_minimal: bool = True,
    out_basename: str | None = None,
    shp_folder_name: str = "shapefile_with_data",
    write_back: bool = True,
    back_layer: str | None = None,
    back_suffix: str = "_with_clusters",
) -> dict:
    """
    Merge clustered labels onto polygons and write a GPKG plus a static PNG map.
    Optionally also write an ESRI Shapefile (.shp).

    Returns
    -------
    dict with keys: gpkg_path, shp_path, png_path, merged_count, writeback_path
    """
    # ---- Locate CSV & labels column robustly ----
    df_path = _find_labels_csv(output_dir, method, k, labels_csv)
    df = pd.read_csv(df_path)

    # normalize 'mukey'
    mukey_candidates = [c for c in df.columns if c.lower() == "mukey"]
    if not mukey_candidates:
        raise KeyError(f"'mukey' column not found in {df_path}")
    if mukey_candidates[0] != "mukey":
        df = df.rename(columns={mukey_candidates[0]: "mukey"})
    df["mukey"] = df["mukey"].astype(str)

    labels_col = _guess_labels_col(df, method, k, labels_col)

    logging.info("Using labels CSV: %s", df_path)
    logging.info("Labels column: %s", labels_col)

    # ---- Load vectors & join ----
    gdf = _load_vector(vector_path)
    gdf = _reproject_and_area(gdf, target_crs=target_crs)

    gdfm = gdf.merge(df[["mukey", labels_col]], on="mukey", how="left")
    merged_count = int(gdfm[labels_col].notna().sum())
    if merged_count == 0:
        logging.warning(
            "No polygons received a label (check MU keys). "
            "mukey examples (vector): %s ; (df): %s",
            gdf["mukey"].head(3).tolist(),
            df["mukey"].head(3).tolist()
        )

    # ---- Prepare output dirs/names ----
    shp_base_dir = os.path.join(output_dir, shp_folder_name)
    _ensure_dir(shp_base_dir)
    base = out_basename if out_basename else f"MO_30cm_clusters_{method}_k{k}"

    gpkg_path = None
    shp_path = None

    # ---- GeoPackage ----
    if write_gpkg:
        gpkg_path = os.path.join(shp_base_dir, f"{base}.gpkg")
        if os.path.exists(gpkg_path):
            try:
                os.remove(gpkg_path)
            except Exception as e:
                logging.warning("Could not remove existing GPKG: %s", e)
        gdfm.to_file(gpkg_path, layer=base, driver="GPKG")
        logging.info("Wrote: %s", gpkg_path)

    # ---- Shapefile (safe fields) ----
    if write_shp:
        shp_dir = os.path.join(shp_base_dir, "shp")
        _ensure_dir(shp_dir)
        shp_path = os.path.join(shp_dir, f"{base}.shp")

        gdf_shp = gdfm.copy()
        gdf_shp["cluster"] = gdf_shp[labels_col]
        keep_cols = ["mukey", "cluster", "geometry"] if shp_minimal else list(gdf_shp.columns)
        gdf_shp = gdf_shp[keep_cols]

        if os.path.exists(shp_path):
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = shp_path.replace(".shp", ext)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        gdf_shp.to_file(shp_path, driver="ESRI Shapefile")
        logging.info("Wrote: %s", shp_path)

    # ---- Quick PNG map ----
    png_path = os.path.join(output_dir, f"map_{method}_k{k}.png")
    try:
        ax = gdfm.plot(column=labels_col, legend=True, cmap=map_cmap, figsize=(10, 8), alpha=0.9, linewidth=0)
        ax.set_axis_off()
        plt.title(f"Clusters — {method} (k={k})")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        logging.info("Wrote: %s", png_path)
    except Exception as e:
        logging.warning("PNG map generation failed: %s", e)
        png_path = None

    # ---- Write back adjacent to source ----
    back_path = None
    if write_back:
        src = Path(vector_path)
        driver = _guess_driver_from_path(vector_path)

        gdf_back = gdfm.copy()
        gdf_back["cluster"] = gdf_back[labels_col]

        if driver == "GPKG":
            back_path = vector_path
            target_layer = back_layer or f"{src.stem}_with_clusters"
            # Write to a new layer; if reusing the same name, this will append in some GDAL stacks
            gdf_back.to_file(back_path, layer=target_layer, driver="GPKG")
            logging.info("Wrote merged layer back into GPKG: %s (layer=%s)", back_path, target_layer)
        elif driver == "ESRI Shapefile":
            back_path = str(src.with_name(f"{src.stem}_with_clusters.shp"))
            gdf_shp_back = gdf_back.copy()
            keep_cols = list(gdf_shp_back.columns)
            if labels_col not in keep_cols:
                keep_cols.append(labels_col)
            gdf_shp_back = gdf_shp_back[keep_cols]
            base_noext = back_path[:-4]
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = base_noext + ext
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            gdf_shp_back.to_file(back_path, driver="ESRI Shapefile")
            logging.info("Wrote merged shapefile next to source: %s", back_path)
        else:
            # Fallback: sibling GPKG
            back_path = str(src.with_name(f"{src.stem}_with_clusters.gpkg"))
            layer = back_layer or f"{src.stem}_with_clusters"
            if os.path.exists(back_path):
                try:
                    os.remove(back_path)
                except Exception as e:
                    logging.warning("Could not remove existing fallback GPKG: %s", e)
            gdf_back.to_file(back_path, layer=layer, driver="GPKG")
            logging.info("Wrote merged GeoPackage (fallback): %s (layer=%s)", back_path, layer)

    return {
        "gpkg_path": gpkg_path,
        "shp_path": shp_path,
        "png_path": png_path,
        "merged_count": merged_count,
        "writeback_path": back_path,
    }


# Convenience wrapper for programmatic use
def make_spatial_products(
    base_dir: str,
    output_dir: str,
    method: str,
    k: int,
    **kwargs,
):
    """
    Convenience wrapper around create_spatial_products.
    By default uses vector_path=f"{base_dir}/mupoly.shp". Override via: vector_path=f"{base_dir}/mupoly.gpkg"
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
                   help="Directory containing clustered CSV (or where it can be auto-discovered).")
    p.add_argument("-m", "--method", required=True,
                   choices=["KMeans", "Agglomerative", "Birch", "GMM", "FuzzyCMeans"],
                   help="Clustering method used.")
    p.add_argument("-k", "--k", type=int, required=True, help="k used in clustering.")
    p.add_argument("--vector-path", required=True,
                   help="Path to MU polygon vector file (e.g., mupoly.shp or .gpkg).")
    p.add_argument("--labels-csv", default=None,
                   help="(Optional) Explicit path to the labels CSV file.")
    p.add_argument("--labels-col", default=None,
                   help="(Optional) Explicit name of the labels column (e.g., KMeans_k10).")
    p.add_argument("--target-crs", default="EPSG:5070",
                   help="Target CRS for reprojection and area calc (default: EPSG:5070).")
    p.add_argument("--no-gpkg", action="store_true", help="Do not write GeoPackage output.")
    p.add_argument("--no-shp", action="store_true", help="Do not write ESRI Shapefile output.")
    p.add_argument("--shp-full", action="store_true",
                   help="Write Shapefile with all fields (may truncate names). Default is minimal fields.")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase verbosity (-v, -vv).")
    p.add_argument("--no-write-back", action="store_true",
                   help="Do not write merged result back to the source dataset.")
    p.add_argument("--back-layer", default=None,
                   help="(GPKG only) Layer name to write merged output into (default: <stem>_with_clusters).")
    p.add_argument("--out-basename", default=None,
                   help="Base name for outputs (overrides MO_30cm_clusters_{method}_k{k}), e.g., 'mupoly_10'.")
    p.add_argument("--shp-folder-name", default="shapefile_with_data",
                   help="Top-level folder (under output-dir) that will contain the 'shp' subfolder. Default: shapefile_with_data")
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
            labels_csv=args.labels_csv,
            labels_col=args.labels_col,
            write_gpkg=(not args.no_gpkg),
            write_shp=(not args.no_shp),
            shp_minimal=(not args.shp_full),
            write_back=(not args.no_write_back),
            back_layer=args.back_layer,
            out_basename=args.out_basename,
            shp_folder_name=args.shp_folder_name,
        )
        if out.get("writeback_path"):
            print(f" - Wrote back to : {out['writeback_path']}")
        print("✅ Spatial mapping completed successfully.")
        print("Results saved in:", args.output_dir)
        if out.get("gpkg_path"):
            print(f" - GeoPackage: {out['gpkg_path']}")
        if out.get("shp_path"):
            print(f" - Shapefile : {out['shp_path']}")
        if out.get("png_path"):
            print(f" - Map PNG   : {out['png_path']}")
        print(f" - Polygons with labels: {out['merged_count']}")
        return 0
    except Exception as e:
        logging.exception("Spatial mapping failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())


