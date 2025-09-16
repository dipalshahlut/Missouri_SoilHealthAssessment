#!/usr/bin/env python3
# horizon_processing.py
"""
Horizon-level processing for SSURGO-style datasets.

Functions
---------
load_horizon_data(horizon_csv, cfrag_csv)
    Load horizon (chorizon) and coarse fragment (cfrag) tables, merge fragment volume.

prepare_horizon_data(hz_df, comp_df, reskinds_by_cokey=None, rock_na_to_0=True)
    Restrict horizons to major components for the study area, attach component metadata,
    normalize types, and perform light cleaning.

aggregate_horizons_depth_slice(hz_df, comp_df, depth_cm, out_dir, tag)
    Aggregate horizon variables to the component level for a given depth slice (e.g., 10/30/100 cm)
    and save the result to CSV.

quality_check_aggregation(comp_agg_df, depth_cm)
    Produce simple QC coverage summaries (per variable, what % component weight has data per MU).

Notes
-----
- This module assumes your upstream filters (e.g., to Missouri) are already applied via `comp_df`.
- Column names follow SSURGO conventions: cokey, chkey, hzdept_r, hzdepb_r, etc.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils import horizon_to_comp  # uses weighted means, kgOrgC_sum, awc_sum internally

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 1) Load
# ---------------------------------------------------------------------

# SoilAnalysis/horizon_processing.py
import pandas as pd
import numpy as np
import os
import logging

from utils import time_this, get_task_logger, wtd_mean, kgOrgC_sum, awc_sum, MUaggregate, MUAggregate_wrapper, \
    reskind_comppct, concat_names, horizon_to_comp


def load_horizon_data(horizon_path, cfrag_path):
    """
    Load horizon (chorizon.csv) and coarse fragment (cfrag.csv) tables,
    normalize column names, attach fragvol_r, ensure awc_r exists (warn if missing),
    coerce key numeric columns, and return a trimmed dataframe.

    Returns
    -------
    pd.DataFrame
        Full merged horizon dataframe with any available columns plus `fragvol_r` summed by chkey.
    """
    
    print("Loading horizon and fragment data...")
    horizon_ = pd.read_csv(horizon_path, na_values=['', ' '])
    cfrag_ = pd.read_csv(cfrag_path, na_values=['', ' '])

    # Select only relevant columns from fragments and merge
    cfrag_df = cfrag_[['fragvol_r','chkey']].copy()
    # Aggregate fragments per horizon key (chkey) if multiple fragments per horizon exist
    cfrag_agg = cfrag_df.groupby('chkey')['fragvol_r'].sum().reset_index()

    # Merge aggregated fragments with horizon data
    horizon_df = pd.merge(horizon_, cfrag_agg, on='chkey', how='left') # Use left join to keep all horizons
    print(f"Loaded and merged {len(horizon_df)} horizon records.")
    print(f"Horizon columns: {horizon_df.columns}")
    return horizon_df

# ---------------------------------------------------------------------
# 2) Prepare / Clean
# ---------------------------------------------------------------------

def prepare_horizon_data(horizon_df, comp_data_mo, reskinds_by_cokey, rockNA_to_0=True):
    """
    Filter horizons to the study area (via comp_df) and attach component metadata.

    Parameters
    ----------
    horizon_df : pd.DataFrame
        Output of `load_horizon_data`.
    comp_df : pd.DataFrame
        Component metadata already filtered to study area (e.g., Missouri).
        Must include: ['cokey','mukey','majcompflag'].
    reskinds_by_cokey : pd.DataFrame or None
        Optional table with ['cokey','reskinds'] summarizing restriction kinds (renamed to 'kind').
    rock_na_to_0 : bool
        If True, set missing fragvol_r to 0 (common pragmatic choice).

    Returns
    -------
    pd.DataFrame
        Horizon rows limited to **major components** in the study area,
        with component attributes attached and cleaned numeric fields.
    """
    print("Preparing horizon data for MO...")
    # Filter horizon_data to include only matching cokeys from comp_data_mo
    horizon_data_mo = horizon_df[horizon_df['cokey'].isin(comp_data_mo['cokey'])].copy()
    print(f"Filtered {len(horizon_data_mo)} horizon records for MO.")

    if horizon_data_mo.empty:
        print("Warning: No horizon data found for the provided component keys.")
        return horizon_data_mo # Return empty DataFrame

    # Assign majcompflag and mukey from comp_data_mo
    cokey_map = comp_data_mo.set_index('cokey')
    horizon_data_mo['majcompflag'] = horizon_data_mo['cokey'].map(cokey_map['majcompflag'])
    horizon_data_mo['mukey'] = horizon_data_mo['cokey'].map(cokey_map['mukey'])

    # Filter only rows where majcompflag is 'Yes' for major component analysis
    horizons_mo_majcomps = horizon_data_mo[horizon_data_mo['majcompflag'] == 'Yes'].copy()
    print(f"Filtered {len(horizons_mo_majcomps)} horizon records for Major Components.")

    if horizons_mo_majcomps.empty:
        print("Warning: No major component horizon data found.")
        return horizons_mo_majcomps

    # Handle missing rock fragment volume ('fragvol_r')
    missing_fragvol = horizons_mo_majcomps['fragvol_r'].isna().sum()
    print(f"Initial missing fragvol_r values in major components: {missing_fragvol}")
    if rockNA_to_0:
        horizons_mo_majcomps['fragvol_r'].fillna(0, inplace=True)
        print(f"Missing fragvol_r values after fillna(0): {horizons_mo_majcomps['fragvol_r'].isna().sum()}")

    # Add restriction 'kind' information
    if reskinds_by_cokey is not None and not reskinds_by_cokey.empty:
         # Ensure cokey types match if needed
         # reskinds_by_cokey['cokey'] = reskinds_by_cokey['cokey'].astype(horizons_mo_majcomps['cokey'].dtype) 
         horizons_mo_majcomps = horizons_mo_majcomps.merge(
            reskinds_by_cokey[['cokey', 'reskinds']].rename(columns={'reskinds': 'kind'}),
            on='cokey', how='left'
         )
         print("Added restriction kinds to horizon data.")
         print(f"Missing restriction kinds after merge: {horizons_mo_majcomps['kind'].isna().sum()}")
    else:
         print("Skipping addition of restriction kinds as reskinds_by_cokey is empty/None.")
         horizons_mo_majcomps['kind'] = None # Add empty column


    # Estimate soil depth (max bottom depth per component)
    # Ensure depths are numeric
    horizons_mo_majcomps['hzdepb_r'] = pd.to_numeric(horizons_mo_majcomps['hzdepb_r'], errors='coerce')
    soil_depths = horizons_mo_majcomps.groupby('cokey')['hzdepb_r'].max().reset_index(name='soil_depth')
    horizons_mo_majcomps = horizons_mo_majcomps.merge(soil_depths, on='cokey', how='left')
    print("Estimated soil depth per component.")

    # --- Optional: Calculate profile-level aggregates if needed before horizon_to_comp ---
    # print("Calculating profile-level weighted means/sums (optional step)...")
    # horizons_mo_majcomps['clay_wtd_mean_profile'] = wtd_mean(horizons_mo_majcomps, 'claytotal_r', weight_col=None, depth_cols=('hzdept_r', 'hzdepb_r')) # Adjust args as needed
    # horizons_mo_majcomps['kgOrg_m2_profile'] = kgOrgC_sum(horizons_mo_majcomps) # Adjust args as needed
    
    print("Horizon data preparation complete.")
    return horizons_mo_majcomps

# ---------------------------------------------------------------------
# 3) Aggregate horizon ➜ component at a depth slice
# ---------------------------------------------------------------------

def aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, depth, output_dir, filename_suffix):
    """
    Aggregate horizon variables to the component level up to `depth_cm` and save to CSV.

    Parameters
    ----------
    hz_df : pd.DataFrame
        Prepared horizon rows (from `prepare_horizon_data`).
    comp_df : pd.DataFrame
        Component metadata (at least ['cokey','mukey','comppct_r']).
    depth_cm : int
        Depth threshold in centimeters (e.g., 10, 30, 100).
    out_dir : str
        Output directory; CSV will be written here.
    filename_suffix : str
        A short label used in the filename (e.g., "30cm").

    Returns
    -------
    pd.DataFrame
        Component-level aggregates with columns like clay_30cm, om_30cm, kgOrg.m2_30cm, awc_30cm, etc.
    """
    if horizons_mo_majcomps.empty:
        print(f"Skipping aggregation for {depth}cm: No major component horizon data.")
        return None

    print(f"Aggregating horizon data to component level for {depth}cm depth...")
    # Ensure comp_data_mo has the necessary columns ('cokey', 'mukey', 'comppct_r')
    required_cols = ['cokey', 'mukey', 'comppct_r']
    if not all(col in comp_data_mo.columns for col in required_cols):
        raise ValueError(f"comp_data_mo is missing one or more required columns: {required_cols}")

    comp_agg = horizon_to_comp(horizons_mo_majcomps, depth=depth, comp_df=comp_data_mo)

    if comp_agg is not None and not comp_agg.empty:
        print(f"Aggregation complete for {depth}cm. Shape: {comp_agg.shape}")
        # Save intermediate result
        csv_path = os.path.join(output_dir, f"comp_MO_{filename_suffix}.csv")
        comp_agg.to_csv(csv_path, index=False)
        print(f"Saved aggregated component data to {csv_path}")

        # --- Post-aggregation processing (Zero handling, QC) ---
        print(f"Performing post-aggregation checks and cleaning for {depth}cm data...")
        # Identify problematic components (optional)
        # problematic_components = {
        #     col: comp_agg.loc[comp_agg[col].isna(), 'compname'].unique()
        #     for col in comp_agg.columns if col not in ['cokey', 'mukey', 'comppct', 'compname'] # Adjust columns to check
        # }
        # if any(problematic_components.values()):
        #      print(f"Problematic components with NAs found for {depth}cm.")

        # Convert specific zero values to NaN
        zero_vars = [f'om_{depth}cm', f'cec_{depth}cm', f'ksat_{depth}cm', f'awc_{depth}cm']
        # Filter list to only include columns that actually exist in the aggregated df
        zero_vars = [var for var in zero_vars if var in comp_agg.columns]

        for var in zero_vars:
            if (comp_agg[var] == 0).any():
                print(f"Converting 0 to NaN for {var}")
                comp_agg[f"{var}_zero"] = np.where(comp_agg[var] == 0, 'Yes', 'No')
                comp_agg.loc[comp_agg[var] == 0, var] = np.nan

        # Fix Ksat for Rock outcrop if ksat column exists
        ksat_col = f'ksat_{depth}cm'
        if ksat_col in comp_agg.columns and 'compname' in comp_agg.columns:
             rock_ksat_mask = (comp_agg['compname'] == 'Rock outcrop') & (comp_agg[ksat_col] > 0)
             if rock_ksat_mask.any():
                  print(f"Converting Ksat to NaN for {rock_ksat_mask.sum()} 'Rock outcrop' components.")
                  comp_agg.loc[rock_ksat_mask, ksat_col] = np.nan

    else:
        print(f"Aggregation for {depth}cm resulted in an empty DataFrame.")
        return None

    print(f"Post-aggregation processing for {depth}cm complete.")
    return comp_agg


# ---------------------------------------------------------------------
# 4) QC summaries
# ---------------------------------------------------------------------

def quality_check_aggregation(comp_agg, depth):
    """
    Quick QC coverage summaries for component aggregates at the given depth.

    For each metric column in a fixed list for this depth, compute per-MU total component percent
    (comppct) that has data. Return mapping metric_name -> DataFrame[['mukey','comppct_tot']].
    """
    if comp_agg is None or comp_agg.empty:
        print(f"Skipping QC for {depth}cm: Aggregated data is empty.")
        return None

    print(f"Performing quality checks on {depth}cm aggregated data...")
    qc_results = {}
    variables_to_check = [f'om_{depth}cm', f'awc_{depth}cm', f'ksat_{depth}cm', f'cec_{depth}cm',
                           f'clay_{depth}cm', f'bd_{depth}cm', f'ec_{depth}cm', f'pH_{depth}cm', f'lep_{depth}cm']
    # Filter list to only include columns that actually exist
    variables_to_check = [var for var in variables_to_check if var in comp_agg.columns]


    for var in variables_to_check:
        filtered_data = comp_agg.dropna(subset=[var])
        if not filtered_data.empty and 'mukey' in filtered_data.columns and 'comppct' in filtered_data.columns:
            comppct_by_mukey = filtered_data.groupby('mukey')['comppct'].sum().reset_index(name='comppct_tot')
            qc_results[var] = comppct_by_mukey
            print(f"QC for {var}:")
            print(comppct_by_mukey['comppct_tot'].describe())
            low_pct_count = (comppct_by_mukey['comppct_tot'] < 70).sum()
            print(f"  -> Mukeys with comppct_tot < 70: {low_pct_count}")
        else:
            print(f"Skipping QC for {var}: Column not found, no data, or missing mukey/comppct.")
            qc_results[var] = None

    print(f"Quality checks for {depth}cm complete.")
    return qc_results





