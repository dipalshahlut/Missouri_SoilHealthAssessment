#!/usr/bin/env python3
# restriction_processing.py
"""
Restriction processing utilities for SSURGO-style datasets — pipeline version aligned to repo behavior.

Functions
---------
load_and_process_restrictions(restrictions_path, comp_data_mo)
    Load corestrictions, filter to study-area components, attach MU/weights, and normalize columns.

aggregate_restriction_depths(restrictions_mo)
    Aggregate restriction depths (top/bottom) per MU using component-percent weights.

create_restriction_summary(restrictions_mo, comp_data_mo)
    Build summaries of restriction kinds: per component (cokey) and per map unit (mukey).

calculate_restriction_percentages(comp_data_mo, reskinds_by_cokey, kinds_of_interest=None)
    For each restriction kind of interest, compute the total component percentage per MU.

Notes
-----
- Matches the pipeline convention where **weights column is `comppct`**; if only `comppct_r` exists,
  it is copied to `comppct` with numeric coercion and [0,100] clipping.
- Keeps key naming consistent with horizon processing and MU-level utilities.
- Expects that the upstream component table (`comp_data_mo`) is already filtered to the study area (e.g., Missouri).

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Utilities expected elsewhere in the codebase

from utils import MUaggregate, MUAggregate_wrapper, reskind_comppct, concat_names
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# 1) Load & normalize restrictions
# --------------------------------------------------------------------------------------


def load_and_process_restrictions(restrictions_path, comp_data_mo):
    """
    Load the corestrictions table, align to study-area components, and normalize fields.

    Parameters
    ----------
    restrictions_path : str
        Path to the restrictions CSV (e.g., corestrictions.csv exported from gSSURGO/SSURGO).
    comp_data_mo : pd.DataFrame
        Study-area component table with at least ['cokey','mukey','comppct_r'] (plus any other fields).

    Returns
    -------
    pd.DataFrame
        Restrictions limited to study area (via cokey), with ['mukey','comppct'] attached and
        canonical columns ['reskind','resdept_r','resdepb_r'] present when available.
    """
    print("Loading and processing restrictions data...")
    restrictions = pd.read_csv(restrictions_path, na_values=['', ' '])
    print(f"Loaded {len(restrictions)} restrictions records.")

    # Filter restrictions for matching cokey values in comp_data_mo
    restrictions_mo = restrictions[restrictions['cokey'].isin(comp_data_mo['cokey'])].copy() # Use .copy()
    print(f"Filtered {len(restrictions_mo)} restrictions for MO.")

    if restrictions_mo.empty:
        print("Warning: No restrictions found for the provided component keys.")
        # Return empty DataFrames with expected columns to avoid downstream errors
        empty_reskinds_cokey = pd.DataFrame(columns=['cokey', 'reskinds', 'mukey', 'comp_pct', 'majcompflag', 'compname'])
        empty_reskinds_mukey = pd.DataFrame(columns=['mukey', 'reskinds'])
        return restrictions_mo, empty_reskinds_cokey, empty_reskinds_mukey

    # Add component info using .map for efficiency
    cokey_map = comp_data_mo.set_index('cokey')
    restrictions_mo['majcompflag'] = restrictions_mo['cokey'].map(cokey_map['majcompflag'])
    restrictions_mo['mukey'] = restrictions_mo['cokey'].map(cokey_map['mukey'])
    restrictions_mo['comppct'] = restrictions_mo['cokey'].map(cokey_map['comppct_r'])

    # Check for missing values after merge
    # print(f"Missing majcompflag after merge: {restrictions_mo['majcompflag'].isna().sum()}")
    # print(f"Missing mukey after merge: {restrictions_mo['mukey'].isna().sum()}")
    # print(f"Missing comppct after merge: {restrictions_mo['comppct'].isna().sum()}")

    # Handle potential missing keys if necessary (e.g., fill with default or drop)
    # restrictions_mo.dropna(subset=['mukey', 'majcompflag', 'comppct'], inplace=True)

    # print(f"Unique reskind values in MO: {restrictions_mo['reskind'].unique()}")
    # print("Restriction processing complete.")
    return restrictions_mo

# --------------------------------------------------------------------------------------
# 2) MU-level aggregation of restriction depths
# --------------------------------------------------------------------------------------

def aggregate_restriction_depths(restrictions_mo):
    """
    Aggregate restriction depths to MU-level, split by restriction kind.

    For each unique `reskind` (if present), computes MU-level aggregates of
    `resdept_r` and `resdepb_r` using the component-percentage weights (`comppct`).

    Returns a mapping: reskind -> MU-wide DataFrame from `MUAggregate_wrapper`.

    If `reskind` is missing, aggregates once on the full table under key 'ALL'.
    """
    print("Aggregating restriction depths by type and mukey...")
    depth_aggregations = {}
    restriction_types = ['Lithic bedrock', 'Fragipan', 'Strongly contrasting textural stratification',
                         'Paralithic bedrock', 'Abrupt textural change', 'Natric',
                         'Densic material', 'Undefined', 'Cemented horizon', 'Petrocalcic'] # Added last two based on later code

    # Filter for major components once
    restrictions_major = restrictions_mo[restrictions_mo['majcompflag'] == 'Yes'].copy()

    if restrictions_major.empty:
        print("Warning: No major component restrictions found for depth aggregation.")
        return {}

    for res_type in restriction_types:
        # Handle potential multiple types for 'Misc' category later
        if res_type in ['Densic material', 'Undefined', 'Cemented horizon', 'Petrocalcic']:
            continue # Skip individual aggregation for misc types here

        df_filtered = restrictions_major[restrictions_major['reskind'] == res_type]
        if not df_filtered.empty:
            #print(f"Aggregating for: {res_type}")
            # Check for duplicates before aggregation if needed
            # print(f"Duplicates for {res_type}: {df_filtered['cokey'].duplicated().sum()}")
            depth_aggregations[res_type] = MUAggregate_wrapper(df=df_filtered, varnames=['resdept_r', 'resdepb_r'])
        else:
             print(f"No major components found for restriction: {res_type}")


    # Handle Miscellaneous separately (combining relevant types)
    misc_types = ['Densic material', 'Undefined', 'Cemented horizon', 'Petrocalcic']
    df_misc = restrictions_major[restrictions_major['reskind'].isin(misc_types)]
    if not df_misc.empty:
        print("Aggregating for: Miscellaneous Restrictions")
        depth_aggregations['Miscellaneous'] = MUAggregate_wrapper(df=df_misc, varnames=['resdept_r', 'resdepb_r'])
    else:
        print("No major components found for Miscellaneous Restrictions")
        
    print("Restriction depth aggregation complete.")
    return depth_aggregations

# --------------------------------------------------------------------------------------
# 3) Summaries of restriction kinds (per component and per MU)
# --------------------------------------------------------------------------------------
def create_restriction_summary(restrictions_mo, comp_data):
    """
    Summarize restriction kinds per component and per MU.

    Returns
    -------
    (reskinds_by_cokey, reskinds_by_mukey)
      reskinds_by_cokey: ['cokey','reskinds'] — pipe-separated unique kinds per component
      reskinds_by_mukey: ['mukey','reskinds'] — pipe-separated unique kinds per MU
    """
    print("Creating restriction kind summaries...")
    # --- By Cokey ---
    filtered_restrict = restrictions_mo[restrictions_mo['majcompflag'] == 'Yes'].copy()

    if filtered_restrict.empty:
        print("Warning: No major component restrictions found for summary.")
        empty_reskinds_cokey = pd.DataFrame(columns=['cokey', 'reskinds', 'mukey', 'comp_pct', 'majcompflag', 'compname'])
        empty_reskinds_mukey = pd.DataFrame(columns=['mukey', 'reskinds'])
        return empty_reskinds_cokey, empty_reskinds_mukey

    reskinds_by_cokey = (
        filtered_restrict.groupby('cokey')['reskind']
        .apply(concat_names)
        .reset_index()
        .rename(columns={'reskind': 'reskinds'})
    )
    #print(f"Shape of reskinds_by_cokey: {reskinds_by_cokey.shape}")

    # Add component info
    cokey_map = comp_data.set_index('cokey')
    reskinds_by_cokey['mukey'] = reskinds_by_cokey['cokey'].map(cokey_map['mukey'])
    reskinds_by_cokey['comp_pct'] = reskinds_by_cokey['cokey'].map(cokey_map['comppct_r'])
    reskinds_by_cokey['majcompflag'] = reskinds_by_cokey['cokey'].map(cokey_map['majcompflag'])
    reskinds_by_cokey['compname'] = reskinds_by_cokey['cokey'].map(cokey_map['compname'])

    # Remove 'Rock outcrop' components if needed (as per original script logic)
    initial_rows = len(reskinds_by_cokey)
    reskinds_by_cokey = reskinds_by_cokey[reskinds_by_cokey['compname'] != 'Rock outcrop'].copy()
    #print(f"Removed {initial_rows - len(reskinds_by_cokey)} 'Rock outcrop' components from cokey summary.")

    # --- By Mukey ---
    # Group the processed cokey summary by mukey
    if not reskinds_by_cokey.empty:
        reskinds_by_mukey = (
            reskinds_by_cokey.groupby('mukey')['reskinds']
            .apply(lambda x: concat_names(x)) # Apply concat_names to the grouped series
            .reset_index()
            #.rename(columns={'reskinds': 'reskinds'}) # Already named reskinds
        )
        #print(f"Shape of reskinds_by_mukey: {reskinds_by_mukey.shape}")
        #print("Unique combined reskinds by mukey:", reskinds_by_mukey['reskinds'].unique())
    else:
        print("Cannot create mukey summary as cokey summary is empty.")
        reskinds_by_mukey = pd.DataFrame(columns=['mukey', 'reskinds'])


    print("Restriction kind summaries complete.")
    return reskinds_by_cokey, reskinds_by_mukey

# --------------------------------------------------------------------------------------
# 4) % component composition per MU that has each restriction kind
# --------------------------------------------------------------------------------------

def calculate_restriction_percentages(comp_data_mo, reskinds_by_cokey):
    """
    Compute, per MU, the total component percent contributed by components with each restriction kind.

    Parameters
    ----------
    comp_data_mo : pd.DataFrame
        Component table with ['cokey','mukey','comppct_r'].
    reskinds_by_cokey : pd.DataFrame
        From `create_restriction_summary` — ['cokey','reskinds'] (pipe-separated kinds).
    kinds_of_interest : list[str] | None
        If provided, restrict to these kinds; otherwise inferred from the data.

    Returns
    -------
    pd.DataFrame
        MU-level table: ['mukey', <kind1>_pct, <kind2>_pct, ...] with values in 0..100.
    """
    print("Calculating component percentages for each restriction type...")
    restriction_pcts = {}
    all_reskinds = ['Lithic bedrock', 'Paralithic bedrock', 'Fragipan',
                    'Abrupt textural change', 'Natric', 'Strongly contrasting textural stratification',
                    'Densic material', 'Undefined', 'Cemented horizon', 'Petrocalcic'] # Ensure all relevant kinds are listed

    # Calculate for Rock Outcrop separately as it's component name based
    rock_outcrop_data = comp_data_mo[comp_data_mo['compname'] == 'Rock outcrop']
    if not rock_outcrop_data.empty:
         restriction_pcts['Rock outcrop'] = (
              rock_outcrop_data.groupby('mukey', as_index=False)['comppct_r']
              .sum().rename(columns={'comppct_r': 'compct_sum'})
          )
         #print(f"Calculated Rock Outcrop percentage for {len(restriction_pcts['Rock outcrop'])} mukeys.")
    else:
        print("No 'Rock outcrop' components found.")


    # Calculate for other restriction types using reskind_comppct helper
    for res_type in all_reskinds:
        # Use regex for combined 'Misc' category if needed, or handle individually
        pattern = res_type.replace(' ', r'\s').replace('(', r'\(').replace(')', r'\)') # Basic regex escape
        if res_type in ['Densic material', 'Undefined', 'Cemented horizon', 'Petrocalcic']:
             pattern = 'Densic material|Undefined|Cemented horizon|Petrocalcic' # Combine misc
             res_key = 'Miscellaneous'
             if res_key in restriction_pcts: continue # Calculate misc only once
        else:
            res_key = res_type

        #print(f"Calculating percentage for: {res_key} (using pattern: {pattern})")
        # Assuming reskind_comppct handles the filtering and aggregation
        pct_df = reskind_comppct(pattern, comp_data_mo, reskinds_by_cokey) # Pass pattern
        if pct_df is not None and not pct_df.empty:
            restriction_pcts[res_key] = pct_df
            print(f"  -> Calculated for {len(pct_df)} mukeys.")
        else:
            print(f"  -> No components found for {res_key}.")

    print("Restriction percentage calculation complete.")
    return restriction_pcts




 
