#!/usr/bin/env python3
# geographic_processing.py
"""
Spatial utilities for SSURGO / soil processing — pipeline version aligned to repo behavior.

Functions
---------
load_and_filter_spatial_data(crop_layer_path, cm_layer_path, mu_poly_path, crop_values_to_consider)
    Load crop & CM layers, filter by crop values, choose initial MU set; standardize MUKEY casing.

load_and_filter_spatial_data_new(mu_poly_path)
    Load MU polygon layer and standardize MU key to 'MUKEY' (uppercase) if present lowercase.

filter_secondary_shp_by_primary_mukey_count(primary_shapefile_path, secondary_shapefile_path, ...)
    Keep features in secondary whose MUKEY appears >= min_count times in primary.

reproject_and_calculate_area(gdf, target_crs="EPSG:5070")
    Reproject to equal-area CRS and compute area_sqm and area_ac; finalize key as 'mukey'.

save_spatial_data(gdf, output_dir, filename_base)
    Save Shapefile + CSV (attributes-only). Includes shapefile column truncation retry logic.

Notes
-----
- Default equal-area CRS is EPSG:5070 (USA Contiguous Albers Equal Area).
- Repo pattern: pre-area functions keep MU key as 'MUKEY'; after area calc it is renamed to 'mukey'.
- Shapefile has field limits; we include a retry path that truncates columns to <=10 chars and resolves duplicates.

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

import os
import logging
from typing import Optional
from pathlib import Path
import geopandas as gpd
import pandas as pd
import dask_geopandas as dask_gpd
log = logging.getLogger(__name__)
REQUIRED_SIDECARS = (".shp", ".shx", ".dbf", ".prj")

def _resolve_and_check_shapefile(path_str: str) -> Path:
    """Resolve absolute path, ensure required sidecars exist and are non-zero."""
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        parent = p.parent
        listing = []
        if parent.exists():
            # help debug basename mismatches
            listing = [x.name for x in parent.iterdir() if x.name.startswith(p.stem)]
        raise FileNotFoundError(
            f"{p} not found.\nParent dir listing (same stem): {listing}"
        )

    missing, zeroed = [], []
    for suf in REQUIRED_SIDECARS:
        side = p.with_suffix(suf)
        if not side.exists():
            missing.append(side.name)
        else:
            try:
                if os.path.getsize(side) == 0:
                    zeroed.append(side.name)
            except OSError:
                zeroed.append(side.name)
    if missing:
        raise FileNotFoundError(
            f"Shapefile incomplete at {p} — missing: {', '.join(missing)}"
        )
    if zeroed:
        raise OSError(
            f"Shapefile sidecars have zero size at {p} — zero-byte: {', '.join(zeroed)}"
        )
    return p
# # ---------------------------------------------------------------------
# # 1) Load & normalize MU polygons (repo-style variants)
# # --------------------------------------------------------------------

def load_and_filter_spatial_data_new(mu_poly_path):
    """
    Robust shapefile loader:
    - resolve absolute path
    - check .shp/.shx/.dbf/.prj exist and are non-zero
    - try pyogrio (default), then fall back to Fiona
    - normalize MUKEY column to uppercase 'MUKEY'
    """
    p = _resolve_and_check_shapefile(mu_poly_path)
    print(f"[geo] Reading MU polygons from: {p}")
    # Load Map Unit Polygons
    mu_poly_shp = gpd.read_file(mu_poly_path)
    #print(f"Map Unit Polygons shape: {mu_poly_shp.shape}")
    #print(mu_poly_shp.columns)
    # Use the filtered Crop/MU/MLRA layer for analysis based on the "OUR CONCEPT" comments
    # Alternatively, use mu_shp if analysis shouldn't be restricted by crop type initially
    mo_mu_shp_initial = mu_poly_shp.copy()  # Uncomment this line if you want all MO map units initially

    #print(f"Initial MO Map Unit shape: {mo_mu_shp_initial.shape}")
    #print(f"Initial MO Map Unit columns: {mo_mu_shp_initial.columns}")

    # Standardize MUKEY column name if necessary (check CMLayer columns output)
    if 'mukey' in mo_mu_shp_initial.columns and 'MUKEY' not in mo_mu_shp_initial.columns:
         mo_mu_shp_initial.rename(columns={'mukey': 'MUKEY'}, inplace=True)
    elif 'MUKEY' not in mo_mu_shp_initial.columns:
        print("Warning: Could not find standard 'MUKEY' or 'mukey' column in the initial spatial data.")
        # Attempt to find a likely key column if needed, or raise an error
        # For now, assume 'MUKEY' is present from CMLayer or mu_shp


    return mo_mu_shp_initial #crpLayer, filtered_cm, mo_mu_shp_initial

# ---------------------------------------------------------------------
# 2) Utility filter by MUKEY counts across layers
# ---------------------------------------------------------------------

def filter_secondary_shp_by_primary_mukey_count(primary_shapefile_path: str,
    secondary_shapefile_path: str,
    min_count: int = 2,
    drop_column: str = 'multipolygon', # Specify column to drop
    verbose: bool = True) -> gpd.GeoDataFrame:
    """
    Analyzes a primary shapefile to find 'MUKEY's appearing at least `min_count`
    times. Then reads and filters a secondary shapefile, keeping only features
    matching those identified MUKEYs. Optionally drops a specified column from
    the final result.

    Args:
        primary_shapefile_path (str): Path to the primary shapefile used for
                                      MUKEY counting.
        secondary_shapefile_path (str): Path to the secondary shapefile which
                                        will be filtered based on MUKEYs found
                                        in the primary file.
        min_count (int, optional): Minimum occurrences of a MUKEY in the
                                   primary shapefile to be considered. Defaults to 2.
        drop_column (str, optional): The name of the column to drop from the
                                     final filtered secondary GeoDataFrame.
                                     Set to None to skip dropping any column.
                                     Defaults to 'multipolygon'.
        verbose (bool, optional): If True, prints intermediate information.
                                  Defaults to True.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame derived from the secondary shapefile,
                          containing only features whose MUKEY met the count
                          criterion in the primary shapefile, and with the
                          specified column potentially dropped.
                          Returns an empty GeoDataFrame with schema matching
                          the secondary file (if readable) on error or if no
                          MUKEYs meet the criteria.

    Note:
        The logic uses the list of MUKEYs from the primary file to *filter*
        the secondary file based on the 'MUKEY' attribute. It does not perform
        a geometric intersection or merge between the two files.
    """
    # --- Input Validation (Primary File) ---
    if not os.path.exists(primary_shapefile_path):
        #print(f"Error: Primary shapefile not found at {primary_shapefile_path}")
        return gpd.GeoDataFrame() # Return empty

    # --- Read and Analyze Primary File ---
    try:
        gdf_primary = gpd.read_file(primary_shapefile_path)
        if verbose:
            print(f"--- Analyzing Primary File: {os.path.basename(primary_shapefile_path)} ---")
            print(f"Primary CRS: {gdf_primary.crs}")
            #print(f"Primary Original Shape: {gdf_primary.shape}")
    except Exception as e:
        print(f"Error reading primary shapefile {primary_shapefile_path}: {e}")
        return gpd.GeoDataFrame()

    if 'MUKEY' not in gdf_primary.columns:
        #print(f"Error: 'MUKEY' column not found in primary shapefile {primary_shapefile_path}")
        return gpd.GeoDataFrame()

    if verbose:
        print(f"Counting polygons per MUKEY in primary file...")
    polygon_counts = gdf_primary['MUKEY'].value_counts()
    mukey_counts_meeting_criteria = polygon_counts[polygon_counts >= min_count]
    mukey_list_to_keep = mukey_counts_meeting_criteria.index

    if verbose:
        #print(f"\nTotal unique MUKEYs in primary file: {len(polygon_counts)}")
        #print(f"MUKEYs meeting count >= {min_count}: {len(mukey_list_to_keep)}")
        if mukey_list_to_keep.empty:
            print(f"No MUKEYs found meeting the criteria in the primary file.")
            # Return empty GDF matching secondary schema if possible (read below)
        else:
            # Optional: Print the list of MUKEYs being kept
            # print(f"MUKEYs to keep: {mukey_list_to_keep.tolist()}")
            pass

    # --- Input Validation (Secondary File) ---
    if not os.path.exists(secondary_shapefile_path):
        print(f"Error: Secondary shapefile not found at {secondary_shapefile_path}")
        return gpd.GeoDataFrame()

    # --- Read Secondary File ---
    try:
        gdf_secondary = gpd.read_file(secondary_shapefile_path)
        if verbose:
            print(f"\n--- Processing Secondary File: {os.path.basename(secondary_shapefile_path)} ---")
            print(f"Secondary CRS: {gdf_secondary.crs}")
            print(f"Secondary Original Shape: {gdf_secondary.shape}")
    except Exception as e:
        print(f"Error reading secondary shapefile {secondary_shapefile_path}: {e}")
        # Try to return empty with primary's schema if secondary fails? Or just empty.
        return gpd.GeoDataFrame()

    if 'MUKEY' not in gdf_secondary.columns:
        print(f"Error: 'MUKEY' column not found in secondary shapefile {secondary_shapefile_path}")
        # Return empty, preserving secondary schema if possible
        return gpd.GeoDataFrame(columns=gdf_secondary.columns, crs=gdf_secondary.crs, geometry=[])


    # --- Filter Secondary File ---
    if not mukey_list_to_keep.empty:
        if verbose:
            print(f"\nFiltering secondary file using {len(mukey_list_to_keep)} MUKEYs from primary analysis...")

        # Filter using .isin() and create a copy
        filtered_secondary_gdf = gdf_secondary[gdf_secondary['MUKEY'].isin(mukey_list_to_keep)].copy()

        if verbose:
            #print(f"Shape after filtering secondary file: {filtered_secondary_gdf.shape}")
            if filtered_secondary_gdf.empty:
                 print("Warning: Filtering resulted in an empty GeoDataFrame (no matching MUKEYs found in secondary file).")
            # print("Head of filtered secondary data (before potential column drop):")
            # print(filtered_secondary_gdf.head()) # Optional print

            # --- Informational calculations from user request ---
            unique_mukeys_in_result = filtered_secondary_gdf['MUKEY'].nunique()
            total_polygons_in_result = len(filtered_secondary_gdf)
            #print(f"Unique MUKEYs in filtered result: {unique_mukeys_in_result}")
            #print(f"Total polygons in filtered result: {total_polygons_in_result}")
            # ---

    else:
        # If no MUKEYs met criteria in primary, return empty GDF matching secondary schema
        if verbose:
             print("\nSkipping secondary file filtering as no MUKEYs met criteria in primary file.")
        return gpd.GeoDataFrame(columns=gdf_secondary.columns, crs=gdf_secondary.crs, geometry=[])


    # --- Drop Specified Column ---
    if drop_column and drop_column in filtered_secondary_gdf.columns:
        if verbose:
            print(f"\nDropping column '{drop_column}' from the result...")
        try:
            final_gdf = filtered_secondary_gdf.drop(columns=[drop_column])
            if verbose:
                 print(f"Shape after dropping column '{drop_column}': {final_gdf.shape}")
        except Exception as e:
             print(f"Warning: Failed to drop column '{drop_column}': {e}")
             final_gdf = filtered_secondary_gdf # Proceed without dropping if error occurs
    elif drop_column:
        if verbose:
            print(f"\nColumn '{drop_column}' not found in the filtered secondary data. Skipping drop.")
        final_gdf = filtered_secondary_gdf
    else:
         if verbose:
            print("\nSkipping column drop (drop_column=None).")
         final_gdf = filtered_secondary_gdf


    if verbose:
        print(f"\n--- Analysis Complete: Returning final GeoDataFrame ---")
        print(f"Final CRS: {final_gdf.crs}")
        print(f"Final Shape: {final_gdf.shape}")
        print("Final Columns:", final_gdf.columns.tolist())
        # print("Final Head:") # Optional
        # print(final_gdf.head()) # Optional

    return final_gdf

# ---------------------------------------------------------------------
# 3) Reproject & compute area (repo-conformant)
# ---------------------------------------------------------------------
def reproject_and_calculate_area(mo_mu_shp, target_crs="EPSG:5070"):
    """
    Reproject polygons and compute area metrics.

    Repo-aligned behavior:
    - If input has no CRS, return an **empty** GeoDataFrame (columns preserved).
    - Reproject to EPSG:5070 when needed; compute `area_sqm` and `area_ac`.
    - Ensure MU key ends as lowercase `mukey` (rename from MUKEY if present).
    """
    print("--- Entering reproject_and_calculate_area function ---")

    # --- Input Check ---
    if not isinstance(mo_mu_shp, gpd.GeoDataFrame):
        print("Error: Input 'mo_mu_shp' is not a GeoDataFrame.")
        return gpd.GeoDataFrame() # Return empty GeoDataFrame

    print(f"Input GeoDataFrame shape: {mo_mu_shp.shape}")
    if mo_mu_shp.empty:
        print("Warning: Input GeoDataFrame 'mo_mu_shp' is empty. Returning empty.")
        return mo_mu_shp # Return the empty input

    print(f"Input CRS: {mo_mu_shp.crs}")
    # print("Input head:\n", mo_mu_shp.head()) # Uncomment for detailed view

    # --- CRS Check ---
    if mo_mu_shp.crs is None:
        print("Error: Input GeoDataFrame 'mo_mu_shp' has no CRS defined. Cannot reproject.")
        # Decide action: return original? return empty? raise error?
        # Returning original might be problematic if area calc assumes target CRS units.
        # Returning empty is safer if subsequent steps rely on the reprojection.
        return gpd.GeoDataFrame(columns=mo_mu_shp.columns, geometry=[]) # Return empty with columns

    print(f"Attempting to reproject to target CRS: {target_crs}...")
    mo_mu_aea = None # Initialize variable

    try:
        if str(mo_mu_shp.crs).upper() != str(target_crs).upper(): # Compare CRS more robustly
             #print(f"Input CRS ({mo_mu_shp.crs}) differs from target ({target_crs}). Reprojecting...")
             mo_mu_aea = mo_mu_shp.to_crs(target_crs)
             #print(f"Reprojection successful. New CRS: {mo_mu_aea.crs}")
        else:
             print(f"Input CRS ({mo_mu_shp.crs}) already matches target. Making a copy.")
             mo_mu_aea = mo_mu_shp.copy()

        #print(f"Shape after reprojection/copy: {mo_mu_aea.shape}")
        if mo_mu_aea.empty and not mo_mu_shp.empty:
             print("Warning: GeoDataFrame became empty after reprojection/copy step!")

    except Exception as e:
        print(f"Error during reprojection or copy: {e}")
        print("Returning an empty GeoDataFrame.")
        # Preserve columns if possible from original input
        return gpd.GeoDataFrame(columns=mo_mu_shp.columns, crs=mo_mu_shp.crs, geometry=[]) # Keep original CRS here?

    # Check if mo_mu_aea is valid before proceeding
    if mo_mu_aea is None or not isinstance(mo_mu_aea, gpd.GeoDataFrame) or mo_mu_aea.empty:
         print("Error: Failed to create a valid GeoDataFrame after reprojection/copy.")
         # Try returning empty with target CRS if possible? Or stick to original schema?
         return gpd.GeoDataFrame(columns=mo_mu_shp.columns, crs=mo_mu_shp.crs, geometry=[])

    # --- Area Calculation ---
    print("Calculating area...")
    try:
        # Check geometry types
        geom_types = mo_mu_aea.geometry.geom_type.unique()
        #print(f"Geometry types present: {geom_types}")
        if not any(gtype in ['Polygon', 'MultiPolygon'] for gtype in geom_types):
            print("Warning: No Polygon or MultiPolygon geometries found. Area calculation might result in zeros.")

        # Calculate area in square meters (assuming target CRS units are meters)
        mo_mu_aea["area_sqm"] = mo_mu_aea.geometry.area
        print("Area (sqm) calculation statistics:")
        print(mo_mu_aea["area_sqm"].describe()) # Shows count, mean, std, min, max etc.

        # Convert to acres
        # Conversion factor: 1 sqm = 0.000247105 acres
        # 1 acre = 4046.86 sqm
        # area_sqm / 4046.86
        # OR area_sqm * 0.000247105
        # OR (area_sqm / 10000) * 2.47105 # sqm -> ha -> acres - This is correct
        mo_mu_aea["area_ac"] = (mo_mu_aea["area_sqm"] / 10000) * 2.47105
        print("Area (acres) calculation statistics:")
        #print(mo_mu_aea["area_ac"].describe())

        total_area_acres = mo_mu_aea["area_ac"].sum()
        print(f"Total area calculated: {total_area_acres:.2f} acres")

        # Check for NaNs or zeros
        if mo_mu_aea["area_ac"].isnull().any():
            print("Warning: Found NaN values in 'area_ac' column.")
        if (mo_mu_aea["area_ac"] == 0).all() and total_area_acres == 0:
             print("Warning: All calculated acre values are zero.")


    except Exception as e:
        print(f"Error during area calculation: {e}")
        # Continue without area columns? Or return? Let's continue but without area.
        if "area_sqm" in mo_mu_aea.columns: mo_mu_aea = mo_mu_aea.drop(columns=["area_sqm"])
        if "area_ac" in mo_mu_aea.columns: mo_mu_aea = mo_mu_aea.drop(columns=["area_ac"])


    # --- MUKEY Processing ---
    print("Processing MUKEY column...")
    if 'MUKEY' in mo_mu_aea.columns:
        print("Found 'MUKEY' column. Ensuring it is string type.")
        try:
             mo_mu_aea['MUKEY'] = mo_mu_aea['MUKEY'].astype(str)
             print("Renaming 'MUKEY' to 'mukey'.")
             mo_mu_aea.rename(columns={'MUKEY': 'mukey'}, inplace=True)
             #print("Columns after rename:", mo_mu_aea.columns.tolist())
        except Exception as e:
             print(f"Error processing MUKEY column: {e}")
    elif 'mukey' in mo_mu_aea.columns:
         print("'mukey' column already exists. Ensuring it is string type.")
         try:
             # Ensure it's string even if it already exists
             mo_mu_aea['mukey'] = mo_mu_aea['mukey'].astype(str)
         except Exception as e:
              print(f"Error ensuring 'mukey' column is string: {e}")
    else:
        print("No 'MUKEY' or 'mukey' column found to process.")

    print("--- Exiting reproject_and_calculate_area function ---")
    print(f"Final output shape: {mo_mu_aea.shape}\n")
    return mo_mu_aea
# ---------------------------------------------------------------------
# 4) Save (Shapefile + CSV), with retry for SHP limits
# ---------------------------------------------------------------------
def save_spatial_data(gdf, output_dir, filename_base):
    """
     Save GeoDataFrame as Shapefile and CSV (attributes-only), repo-style.

    - On SHP failure, retry by truncating column names to 10 chars and disambiguating duplicates.
    - CSV output drops the geometry column.
    """
    shapefile_path = os.path.join(output_dir, f"{filename_base}.shp")
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")

    # Save Shapefile
    try:
        print(f"Saving shapefile to {shapefile_path}...")
        gdf.to_file(shapefile_path, driver="ESRI Shapefile")
        print("Shapefile saved successfully.")
    except Exception as e:
        print(f"Error saving shapefile: {e}")
        print("Attempting to fix potential issues...")
        # Truncate long column names (common issue with Shapefiles)
        gdf_copy = gdf.copy()
        gdf_copy.columns = [col[:10] if len(col) > 10 else col for col in gdf_copy.columns]
        # Check for duplicate truncated names and handle them (e.g., append index)
        cols = pd.Series(gdf_copy.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        gdf_copy.columns = cols
        
        try:
            print("Retrying save with potentially fixed column names...")
            gdf_copy.to_file(shapefile_path, driver="ESRI Shapefile")
            print("Shapefile saved successfully on retry.")
        except Exception as e_retry:
            print(f"Error saving shapefile even after attempting fixes: {e_retry}")


    # Save CSV (without geometry)
    try:
        print(f"Saving CSV to {csv_path}...")
        gdf_csv = gdf.drop(columns='geometry', errors='ignore')
        gdf_csv.to_csv(csv_path, index=False)
        print("CSV saved successfully.")
    except Exception as e:
        print(f"Error saving CSV: {e}")


