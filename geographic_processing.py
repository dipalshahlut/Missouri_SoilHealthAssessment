#!/usr/bin/env python3
# geographic_processing.py
"""
Spatial utilities for SSURGO / soil processing.

Functions
---------
load_and_filter_spatial_data_new(mu_poly_path)
    Load MU polygon layer (Shapefile/GeoPackage), normalize MU key to 'mukey' (str).

reproject_and_calculate_area(gdf, target_crs="EPSG:5070")
    Reproject polygons to an equal-area CRS and compute area in m² and acres.

save_spatial_data(gdf, out_dir, base_name, layer=None, write_shapefile=False, write_csv=True)
    Save a GeoPackage (recommended), and optionally a Shapefile + CSV export.

Notes
-----
- The default equal-area CRS is **EPSG:5070** (USA Contiguous Albers Equal Area).
- The join key is normalized to **'mukey'** as string for consistency with tabular merges.
- Shapefile has column name & type limitations; GeoPackage (.gpkg) is preferred.
"""


import logging
import os
from typing import Optional

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 1) Load & normalize MU polygons
# ---------------------------------------------------------------------

def load_and_filter_spatial_data_new(mu_poly_path: str) -> gpd.GeoDataFrame:
    """
    Load MU polygons and normalize MU key field.

    Parameters
    ----------
    mu_poly_path : str
        Path to MU polygon dataset (e.g., 'mupoly.shp' or 'mupoly.gpkg').

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with:
          - Geometry column intact
          - A string column 'mukey' (normalized from 'MUKEY' or similar)
          - No other changes (no reprojection/area yet)

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file is empty or has no valid geometries.
    KeyError
        If no MU key field can be identified.
    """
    if not os.path.exists(mu_poly_path):
        raise FileNotFoundError(f"Vector file not found: {mu_poly_path}")

    gdf = gpd.read_file(mu_poly_path)
    if gdf.empty:
        raise ValueError(f"Vector file loaded but empty: {mu_poly_path}")

    if "geometry" not in gdf.columns or gdf.geometry.isna().all():
        raise ValueError("Input vector has no geometry column or all geometries are missing.")

    # Normalize MU key name → 'mukey'
    lower_cols = {c.lower(): c for c in gdf.columns}
    if "mukey" in lower_cols:
        src = lower_cols["mukey"]
        if src != "mukey":
            gdf = gdf.rename(columns={src: "mukey"})
    elif "mukey" in gdf.columns:
        # already correct
        pass
    elif "MUKEY" in gdf.columns:
        gdf = gdf.rename(columns={"MUKEY": "mukey"})
    else:
        # Best-effort: look for a near-match
        candidates = [c for c in gdf.columns if c.lower() == "mukey"]
        if candidates:
            gdf = gdf.rename(columns={candidates[0]: "mukey"})
        else:
            raise KeyError(
                "Could not find a MU key column. Expected 'mukey' or 'MUKEY' in the vector data."
            )

    # Ensure type: string (important for stable joins)
    gdf["mukey"] = gdf["mukey"].astype(str)

    # Basic geometry sanity check
    if gdf.geometry.is_empty.any():
        log.warning("Some geometries are empty; they will be preserved as-is.")

    log.info("Loaded polygons: %s rows; columns: %s", len(gdf), list(gdf.columns))
    return gdf


# ---------------------------------------------------------------------
# 2) Reproject & compute area
# ---------------------------------------------------------------------

def reproject_and_calculate_area(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:5070"
) -> gpd.GeoDataFrame:
    """
    Reproject polygons and compute area metrics.

    - If `gdf.crs` is missing, assumes EPSG:4326 (WGS84) before reprojection.
    - Adds:
        * area_m2 : polygon area in square meters (float)
        * area_ac : polygon area in acres (float)

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input polygons with geometry and 'mukey'.
    target_crs : str
        Target CRS (default 'EPSG:5070' – USA Contiguous Albers Equal Area).

    Returns
    -------
    gpd.GeoDataFrame
        A **copy** of `gdf` reprojected to `target_crs` with area fields appended.
    """
    if gdf.crs is None:
        log.warning("Input vector has no CRS; assuming EPSG:4326 (WGS84) before reprojection.")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    out = gdf.to_crs(target_crs) if str(gdf.crs) != target_crs else gdf.copy()

    # Compute areas (EPSG:5070 units are meters)
    out["area_m2"] = out.geometry.area
    out["area_ac"] = out["area_m2"] * 0.000247105381  # meters² → acres

    log.info(
        "Reprojected to %s; added area_m2 & area_ac. Example area_ac stats: min=%.3f, mean=%.3f, max=%.3f",
        target_crs,
        float(out["area_ac"].min()),
        float(out["area_ac"].mean()),
        float(out["area_ac"].max()),
    )
    return out


# ---------------------------------------------------------------------
# 3) Save
# ---------------------------------------------------------------------

def save_spatial_data(
    gdf: gpd.GeoDataFrame,
    out_dir: str,
    base_name: str,
    layer: Optional[str] = None,
    write_shapefile: bool = False,
    write_csv: bool = True,
) -> dict:
    """
    Save a spatial dataset to disk in robust formats.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Polygons with attributes.
    out_dir : str
        Output directory to create/use.
    base_name : str
        Base name for exported files (e.g., 'MO_30cm_clusters_KMeans_k10').
    layer : str | None
        Layer name for GeoPackage; defaults to base_name.
    write_shapefile : bool
        Also write a Shapefile (.shp). Beware of column name/type limits.
    write_csv : bool
        Also write a CSV (attributes only, no geometry).

    Returns
    -------
    dict
        Paths to written files: {'gpkg': <path>, 'shp': <path or None>, 'csv': <path or None>}

    Notes
    -----
    - GeoPackage is the preferred format (no .dbf limits, preserves types).
    - If Shapefile is requested, long column names may be truncated by the driver.
    """
    os.makedirs(out_dir, exist_ok=True)
    layer_name = layer or base_name

    # 1) GeoPackage
    gpkg_path = os.path.join(out_dir, f"{base_name}.gpkg")
    # If file exists, overwrite the file by removing it (simplest to avoid layer conflicts)
    if os.path.exists(gpkg_path):
        try:
            os.remove(gpkg_path)
        except Exception as e:
            log.warning("Could not remove existing GPKG; will overwrite layer inside: %s", e)

    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    log.info("Wrote GeoPackage: %s (layer=%s)", gpkg_path, layer_name)

    # 2) Optional Shapefile
    shp_path = None
    if write_shapefile:
        shp_path = os.path.join(out_dir, f"{base_name}.shp")
        # Shapefile cannot overwrite multiple files atomically; let driver handle replace
        try:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            log.info("Wrote Shapefile: %s", shp_path)
        except Exception as e:
            log.warning("Writing Shapefile failed (field/type limits?): %s", e)
            shp_path = None

    # 3) Optional CSV (attributes only)
    csv_path = None
    if write_csv:
        csv_path = os.path.join(out_dir, f"{base_name}.csv")
        try:
            # Drop geometry for tabular CSV
            attrs = pd.DataFrame(gdf.drop(columns="geometry"))
            attrs.to_csv(csv_path, index=False)
            log.info("Wrote attributes CSV: %s", csv_path)
        except Exception as e:
            log.warning("Writing CSV failed: %s", e)
            csv_path = None

    return {"gpkg": gpkg_path, "shp": shp_path, "csv": csv_path}
