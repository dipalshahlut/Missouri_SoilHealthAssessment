#!/usr/bin/env python3
"""
main.py — repo-aligned pipeline with a small CLI wrapper.

Replicates the uploaded main.py behavior:
  1) Load MU polygons, reproject to EPSG:5070, compute area, MUKEY→mukey
  2) Load mapunit/component/muagg, right-join on mukey, filter to spatial MUKEYs
  3) Component summaries and flags
  4) Restrictions: depths, summaries, percentages
  5) Horizons: load/prepare, aggregate 10/30/100 cm, QC
  6) Integrate all to polygons
  7) Write shapefile+CSV of polygons and the final analysis CSV

Usage:
python main.py \
  --base-dir /path/to/data \
  --output-dir /path/to/data/aggResult \
  --target-crs EPSG:5070 \
  --analysis-depth 30
  
__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations
import os, time, logging, argparse, sys
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

# --- project modules (unchanged) ---
from geographic_processing import (
    load_and_filter_spatial_data_new,
    reproject_and_calculate_area,
    save_spatial_data,
)
from restriction_processing import (
    load_and_process_restrictions,
    aggregate_restriction_depths,
    create_restriction_summary,
    calculate_restriction_percentages,
)
from horizon_processing import (
    load_horizon_data,
    prepare_horizon_data,
    aggregate_horizons_depth_slice,
    quality_check_aggregation,
)
from utils import concat_names, MUAggregate_wrapper

# ---------------- configuration ----------------

ASSUMED_DEPTH    = 200         # cm
ROCK_NA_TO_0     = True


# Get a logger specific to main
log = logging.getLogger(__name__) 

# --- Main Workflow ---

def main(argv=None):
    args = build_argparser().parse_args(argv)

    # Use CLI-supplied values (fall back to defaults inside the parser)
    BASE_DIR       = args.base_dir
    OUTPUT_DIR     = args.output_dir
    TARGET_CRS     = args.target_crs
    ANALYSIS_DEPTH = args.analysis_depth

    MU_COUNTY_SHP = os.path.join(BASE_DIR, "MO_County_Boundaries.shp") 
    MU_POLY_PATH = os.path.join(BASE_DIR, "mupoly.shp")
    MAPUNIT_CSV_PATH = os.path.join(BASE_DIR, "mapunit.csv")
    COMPONENT_CSV_PATH = os.path.join(BASE_DIR, "component.csv")
    MUAGG_CSV_PATH = os.path.join(BASE_DIR, "muagg.csv")
    RESTRICTIONS_CSV_PATH = os.path.join(BASE_DIR, "corestrictions.csv")
    HORIZON_CSV_PATH = os.path.join(BASE_DIR, "chorizon.csv")
    CFRAG_CSV_PATH = os.path.join(BASE_DIR, "cfrag.csv")

    log.info("Starting main workflow...")
    overall_start_time = time.time() #  time the whole workflow
    print("Starting SSURGO Data Processing Workflow...")
    # --- THE TRY BLOCK WRAPS YOUR ENTIRE WORKFLOW ---
    try:

        # --- 1. Load and Prepare Initial Data ---
        log.info("--- Workflow Step: Geographic Processing ---")
        print("\n--- Step 1: Loading Initial Data ---")
        # Load spatial data
    
        mo_mu_shp_initial = load_and_filter_spatial_data_new( MU_POLY_PATH)

        
        # Reproject and calculate area
        mo_mu_shp = reproject_and_calculate_area(mo_mu_shp_initial, TARGET_CRS)
        print(f"Reproject MU data for MO: {mo_mu_shp.shape}")   
        # Load tabular data
        mu_data = pd.read_csv(MAPUNIT_CSV_PATH, na_values=['', ' '])
        # Ensure DataFrame column is string
        mu_data['mukey'] = mu_data['mukey'].astype(str)
        comp_ = pd.read_csv(COMPONENT_CSV_PATH, na_values=['', ' '])
        map_df = pd.read_csv(MUAGG_CSV_PATH, na_values=['', ' ']) # muaggatt data

        # Merge component and muaggatt
        comp_df = pd.merge(comp_, map_df, on='mukey', how='right') # Right join keeps all muaggatt mukeys
        # Ensure DataFrame column is string
        comp_df['mukey'] = comp_df['mukey'].astype(str)

        # Filter tabular data for MO mukeys present in the spatial data
        mukeys_in_spatial = mo_mu_shp['mukey'].unique()
        print(f"Unique MUKEY for MO are: {mukeys_in_spatial.shape}")
        mu_data_mo = mu_data[mu_data['mukey'].isin(mukeys_in_spatial)].copy()
        print(f"Filtered Map Unit Data for MO: {mu_data_mo.shape}")
        comp_data_mo = comp_df[comp_df['mukey'].isin(mukeys_in_spatial)].copy()
        print(f"Filtered component data for MO: {comp_data_mo.shape}")


        # --- 2. Process Component Data ---
        log.info("--- Workflow Step: Component data Processing ---")
        print("\n--- Step 2: Processing Component Data ---")
        # Check and fix component data issues (majcompflag, NAs)
        print("Checking component data for NA values...")
        if comp_data_mo['majcompflag'].isna().any():
            print("Warning: NAs found in 'majcompflag'. Review data or handle NAs.")
            # comp_data_mo.dropna(subset=['majcompflag'], inplace=True) # Example: drop NAs
        if comp_data_mo['comppct_r'].isna().any():
            print("Warning: NAs found in 'comppct_r'. Dropping these rows.")
            comp_data_mo.dropna(subset=['comppct_r'], inplace=True)

        # Fix majcompflag errors (as per original logic) - Be cautious with this logic
        print("Applying majcompflag corrections based on comppct_r...")
        comp_data_mo.loc[(comp_data_mo['majcompflag'] == 'No ') & (comp_data_mo['comppct_r'] >= 15), 'majcompflag'] = 'Yes'
        comp_data_mo.loc[(comp_data_mo['majcompflag'] == 'Yes') & (comp_data_mo['comppct_r'] < 15), 'majcompflag'] = 'No '
        print("'majcompflag' corrections applied.")

        # Create component summaries by mukey
        majcomps_no_by_mukey = (comp_data_mo[comp_data_mo['majcompflag'] == 'Yes']
                                .groupby('mukey').size().reset_index(name='majcomp_no'))
        majcompnames_by_mukey = (comp_data_mo[comp_data_mo['majcompflag'] == 'Yes']
                                .groupby('mukey')['compname'].apply(concat_names)
                                .reset_index().rename(columns={'compname': 'majcompnames'}))
        majcomp_taxorders_by_mukey = (comp_data_mo[comp_data_mo['majcompflag'] == 'Yes']
                                    .groupby('mukey')['taxorder'].apply(concat_names)
                                    .reset_index().rename(columns={'taxorder': 'taxorders'}))
        compnames_by_mukey = (comp_data_mo.groupby('mukey')['compname']
                            .apply(concat_names).reset_index().rename(columns={'compname': 'compnames'}))
        domcomp_pct_by_mukey = (comp_data_mo.groupby('mukey')['comppct_r']
                                .max().reset_index().rename(columns={'comppct_r': 'docomppct'}))
        majcomp_pct_by_mukey = (comp_data_mo[comp_data_mo['majcompflag'] == 'Yes']
                                .groupby('mukey')['comppct_r'].sum()
                                .reset_index().rename(columns={'comppct_r': 'majcomppct'}))
        print("Generated component summaries by mukey.")


        # --- 3. Process Restrictions ---
        log.info("--- Workflow Step: Restriction Processing ---")
        print("\n--- Step 3: Processing Restrictions ---")
        restrictions_mo = load_and_process_restrictions(RESTRICTIONS_CSV_PATH, comp_data_mo)
        
        # Only proceed if restrictions were found
        restriction_depths_agg = {}
        restriction_pcts_agg = {}
        if not restrictions_mo.empty:
            restriction_depths_agg = aggregate_restriction_depths(restrictions_mo)
            
            # Create cokey/mukey restriction summaries needed for percentage calculation
            reskinds_by_cokey_summary, reskinds_by_mukey_summary = create_restriction_summary(restrictions_mo, comp_data_mo)
            
            # Calculate component percentages for each restriction type
            restriction_pcts_agg = calculate_restriction_percentages(comp_data_mo, reskinds_by_cokey_summary) # Pass summary df
        else:
            print("Skipping restriction depth aggregation and percentage calculation as no MO restrictions were loaded.")
            reskinds_by_mukey_summary = pd.DataFrame(columns=['mukey', 'reskinds']) # Ensure exists even if empty


        # --- 4. Process Horizons ---
        log.info("--- Workflow Step: Horizon Processing ---")
        print("\n--- Step 4: Processing Horizons ---")
        horizon_df = load_horizon_data(HORIZON_CSV_PATH, CFRAG_CSV_PATH)
        horizons_mo_majcomps = prepare_horizon_data(horizon_df, comp_data_mo, reskinds_by_cokey_summary, ROCK_NA_TO_0) # Pass summary df

        # Aggregate horizon data for different depth slices
        comp_mo_10cm = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 10, OUTPUT_DIR, "10cm")
        comp_mo_30cm = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 30, OUTPUT_DIR, "30cm")
        comp_mo_100cm = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 100, OUTPUT_DIR, "100cm")

        # Perform QC on aggregated data (comppct checks)
        qc_results_10cm = quality_check_aggregation(comp_mo_10cm, 10)
        qc_results_30cm = quality_check_aggregation(comp_mo_30cm, 30)
        qc_results_100cm = quality_check_aggregation(comp_mo_100cm, 100)


        # --- 5. Integrate Data into Map Unit Polygons ---
        print("\n--- Step 5: Integrating Data into Map Unit Polygons ---")
        
        # Ensure mukey types match for merging (should be string after reproject_and_calculate_area)
        mo_mu_shp['mukey'] = mo_mu_shp['mukey'].astype(str)
        print(f"The Mo_mu_shp shape is : {mo_mu_shp.shape}")
        # Add component summary info
        print("Adding component summaries to spatial data...")
        # Convert keys to string in summary dfs before mapping
        majcomps_no_by_mukey['mukey'] = majcomps_no_by_mukey['mukey'].astype(str)
        majcomp_taxorders_by_mukey['mukey'] = majcomp_taxorders_by_mukey['mukey'].astype(str)
        domcomp_pct_by_mukey['mukey'] = domcomp_pct_by_mukey['mukey'].astype(str)
        majcomp_pct_by_mukey['mukey'] = majcomp_pct_by_mukey['mukey'].astype(str)
        majcompnames_by_mukey['mukey'] = majcompnames_by_mukey['mukey'].astype(str)
        compnames_by_mukey['mukey'] = compnames_by_mukey['mukey'].astype(str)
        mu_data_mo['mukey'] = mu_data_mo['mukey'].astype(str)

        mo_mu_shp['mjcps_no'] = mo_mu_shp['mukey'].map(majcomps_no_by_mukey.set_index('mukey')['majcomp_no'])
        mo_mu_shp['txorders'] = mo_mu_shp['mukey'].map(majcomp_taxorders_by_mukey.set_index('mukey')['taxorders'])
        mo_mu_shp['dmcmp_pct'] = mo_mu_shp['mukey'].map(domcomp_pct_by_mukey.set_index('mukey')['docomppct'])
        mo_mu_shp['mjcmp_pct'] = mo_mu_shp['mukey'].map(majcomp_pct_by_mukey.set_index('mukey')['majcomppct'])
        mo_mu_shp['mjcmpnms'] = mo_mu_shp['mukey'].map(majcompnames_by_mukey.set_index('mukey')['majcompnames'])
        mo_mu_shp['compnames'] = mo_mu_shp['mukey'].map(compnames_by_mukey.set_index('mukey')['compnames'])
        mo_mu_shp['muname'] = mo_mu_shp['mukey'].map(mu_data_mo.set_index('mukey')['muname'])
        mo_mu_shp['complex'] = mo_mu_shp['muname'].str.contains('complex', case=False, na=False).map({True: 'Yes', False: 'No'})
        mo_mu_shp['associan'] = mo_mu_shp['muname'].str.contains('association', case=False, na=False).map({True: 'Yes', False: 'No'})


        # Add restriction summary and individual flags
        print("Adding restriction summaries and flags...")
        if not reskinds_by_mukey_summary.empty:
            reskinds_by_mukey_summary['mukey'] = reskinds_by_mukey_summary['mukey'].astype(str)
            mo_mu_shp['restrict'] = mo_mu_shp['mukey'].map(reskinds_by_mukey_summary.set_index('mukey')['reskinds'])
            mo_mu_shp['restrict'].fillna('None', inplace=True)
            
            # Add flags based on the 'restrict' column content
            mo_mu_shp['Lithic'] = mo_mu_shp['restrict'].str.contains('Lithic bedrock', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Paralith'] = mo_mu_shp['restrict'].str.contains('Paralithic bedrock', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Fragipan'] = mo_mu_shp['restrict'].str.contains('Fragipan', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['ATC'] = mo_mu_shp['restrict'].str.contains('Abrupt textural change', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Natric'] = mo_mu_shp['restrict'].str.contains('Natric', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['SCTS'] = mo_mu_shp['restrict'].str.contains('Strongly contrasting textural stratification', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Misc_Res'] = mo_mu_shp['restrict'].str.contains('Densic material|Cemented horizon|Petrocalcic|Undefined', na=False).map({True: 'Yes', False: 'No'}) # Added Undefined
        else:
            print("Skipping restriction summary merge as it was empty.")
            mo_mu_shp['restrict'] = 'None'
            for flag in ['Lithic', 'Paralith', 'Fragipan', 'ATC', 'Natric', 'SCTS', 'Misc_Res']:
                mo_mu_shp[flag] = 'No'

        # Add Rock Outcrop flag based on component names
        mo_mu_shp['Rock_OC'] = mo_mu_shp['compnames'].str.contains('Rock outcrop', na=False).map({True: 'Yes', False: 'No'})
        print(f"MU polygon shape file shape is : {mo_mu_shp.shape}")
        # Add restriction depths
        print("Adding restriction depths...")
        # Map restriction depths using the aggregated dictionaries
        depth_mapping = {
            'Lithic': ('Lthc_dep', 'Lithic bedrock'),
            'Paralith': ('Plth_dep', 'Paralithic bedrock'),
            'Fragipan': ('Frpn_dep', 'Fragipan'),
            'ATC': ('ATC_dep', 'Abrupt textural change'),
            'Natric': ('Natr_dep', 'Natric'),
            'SCTS': ('SCTS_dep', 'Strongly contrasting textural stratification'),
            'Misc_Res': ('MRes_dep', 'Miscellaneous') # Key used in aggregate_restriction_depths
        }
        depth_cols_added = []
        for flag_col, (depth_col, res_key) in depth_mapping.items():
            mo_mu_shp[depth_col] = ASSUMED_DEPTH # Default
            if res_key in restriction_depths_agg and not restriction_depths_agg[res_key].empty:
                depth_df = restriction_depths_agg[res_key]
                depth_df['mukey'] = depth_df['mukey'].astype(str)
                # Use resdept_r (top depth)
                mo_mu_shp[depth_col] = mo_mu_shp.apply(
                    lambda row: depth_df.set_index('mukey').loc[row['mukey']]['resdept_r'] if row[flag_col] == 'Yes' and row['mukey'] in depth_df['mukey'].values else row[depth_col],
                    axis=1
                )
                depth_cols_added.append(depth_col)
            else:
                print(f"No aggregated depth data found for {res_key}.")
                
        # Calculate Minimum Restriction Depth
        if depth_cols_added:
            mo_mu_shp['MnRs_dep'] = mo_mu_shp[depth_cols_added].min(axis=1)
            print("Calculated minimum restriction depth.")
            print(mo_mu_shp[['MnRs_dep'] + depth_cols_added].describe())
        else:
            print("Minimum restriction depth calculation skipped as no depth columns were added.")
            mo_mu_shp['MnRs_dep'] = ASSUMED_DEPTH


        # Add restriction percentages
        print("Adding restriction percentages...")
        pct_mapping = {
            'Lithic': ('Lthc_pct', 'Lithic bedrock'),
            'Rock_OC':('RckOC_pct','Rock outcrop'), # Special case handled by name
            'Paralith': ('Plth_pct', 'Paralithic bedrock'),
            'Fragipan': ('Frpn_pct', 'Fragipan'),
            'ATC': ('ATC_pct', 'Abrupt textural change'),
            'Natric': ('Natr_pct', 'Natric'),
            'SCTS': ('SCTS_pct', 'Strongly contrasting textural stratification'),
            'Misc_Res': ('MRes_pct', 'Miscellaneous') # Key used in calculate_restriction_percentages
        }
        pct_cols_added = []
        for flag_col, (pct_col, res_key) in pct_mapping.items():
            mo_mu_shp[pct_col] = 0.0 # Default to 0
            if res_key in restriction_pcts_agg and not restriction_pcts_agg[res_key].empty:
                pct_df = restriction_pcts_agg[res_key]
                pct_df['mukey'] = pct_df['mukey'].astype(str)
                # Map the compct_sum
                mo_mu_shp[pct_col] = mo_mu_shp.apply(
                    lambda row: pct_df.set_index('mukey').loc[row['mukey']]['compct_sum'] if row[flag_col] == 'Yes' and row['mukey'] in pct_df['mukey'].values else row[pct_col],
                    axis=1
                )
                pct_cols_added.append(pct_col)
            else:
                print(f"No aggregated percentage data found for {res_key}.")
                
        if pct_cols_added:
            print(mo_mu_shp[pct_cols_added].describe())
        else:
            print("Restriction percentage merge skipped.")

        # Add aggregated horizon data (using the chosen ANALYSIS_DEPTH)
        print(f"Adding aggregated horizon data for {ANALYSIS_DEPTH}cm...")
        comp_agg_dict = {10: comp_mo_10cm, 30: comp_mo_30cm, 100: comp_mo_100cm}
        comp_analysis_agg = comp_agg_dict.get(ANALYSIS_DEPTH)

        if comp_analysis_agg is not None and not comp_analysis_agg.empty:
            # Aggregate component data to MU level using MUAggregate_wrapper from utils
            from utils import MUAggregate_wrapper # Ensure import
            
            # Define varnames based on the aggregated component dataframe columns
            # Exclude non-numeric/ID columns like cokey, mukey, compname, comppct, flags etc.
            cols_to_exclude = ['cokey', 'mukey', 'compname', 'comppct'] + [col for col in comp_analysis_agg.columns if '_zero' in col]
            varnames_agg = [col for col in comp_analysis_agg.columns if col not in cols_to_exclude]
            
            print(f"Aggregating variables to mapunit level: {varnames_agg}")
            mu_analysis_agg = MUAggregate_wrapper(df=comp_analysis_agg, varnames=varnames_agg)

            if mu_analysis_agg is not None and not mu_analysis_agg.empty:
                mu_analysis_agg['mukey'] = mu_analysis_agg['mukey'].astype(str)
                # Merge with the main spatial dataframe
                mo_mu_shp = mo_mu_shp.merge(mu_analysis_agg, on='mukey', how='left')
                print(f"Merged {ANALYSIS_DEPTH}cm aggregated horizon data. New shape: {mo_mu_shp.shape}")
                # Check merge result - how many NAs introduced?
                print(f"NA count for first merged column ({varnames_agg[0]}): {mo_mu_shp[varnames_agg[0]].isna().sum()}")
            else:
                print(f"Mapunit aggregation failed for {ANALYSIS_DEPTH}cm data.")
        else:
            print(f"Skipping merge of {ANALYSIS_DEPTH}cm horizon data as it's empty/None.")


        # Add QC metrics (component percentage coverage for key variables)
        print("Adding QC metrics (component coverage)...")
        qc_results_dict = {10: qc_results_10cm, 30: qc_results_30cm, 100: qc_results_100cm}
        qc_analysis = qc_results_dict.get(ANALYSIS_DEPTH)
        if qc_analysis:
            qc_vars_mapping = {
                f'om_{ANALYSIS_DEPTH}cm': 'compct_om',
                f'cec_{ANALYSIS_DEPTH}cm': 'compct_cec',
                f'ksat_{ANALYSIS_DEPTH}cm': 'compct_ksat',
                f'awc_{ANALYSIS_DEPTH}cm': 'compct_awc',
                f'clay_{ANALYSIS_DEPTH}cm': 'compct_clay',
                f'bd_{ANALYSIS_DEPTH}cm': 'compct_bd',
                f'ec_{ANALYSIS_DEPTH}cm': 'compct_ec', # Check if EC was aggregated
                f'pH_{ANALYSIS_DEPTH}cm': 'compct_pH',
                f'lep_{ANALYSIS_DEPTH}cm': 'compct_lep',
            }
            for qc_var, target_col in qc_vars_mapping.items():
                if qc_analysis.get(qc_var) is not None and not qc_analysis[qc_var].empty:
                    qc_df = qc_analysis[qc_var]
                    qc_df['mukey'] = qc_df['mukey'].astype(str)
                    mo_mu_shp[target_col] = mo_mu_shp['mukey'].map(qc_df.set_index('mukey')['comppct_tot'])
                    print(f"Added QC column: {target_col}")
                else:
                    print(f"Skipping QC column {target_col} as source data was missing.")
                    mo_mu_shp[target_col] = np.nan # Assign NaN if QC data doesn't exist
        else:
            print(f"Skipping addition of QC metrics as QC results for depth {ANALYSIS_DEPTH}cm are missing.")

        print(f"The MU polygon file shape before clustering is: {mo_mu_shp.shape}")
        # --- 6. Final Preparation and QC before Clustering ---
        log.info("--- Workflow Step: Final data Processing ---")
        print("\n--- Step 6: Final Preparation and QC ---")
        
        # Aggregate area by mukey and remove duplicate polygons, keeping one per mukey
        acres_by_mukey = mo_mu_shp.groupby('mukey', as_index=False)['area_ac'].sum()
        print(acres_by_mukey)
        mo_mu_agg_by_mukey = mo_mu_shp.drop_duplicates(subset='mukey', keep='first').copy()
        # Map the summed area back
        mo_mu_agg_by_mukey.drop(columns=['area_ac'], inplace=True, errors='ignore') # Drop original area if exists
        mo_mu_agg_by_mukey = mo_mu_agg_by_mukey.merge(acres_by_mukey, on='mukey', how='left')
        print(f"Aggregated spatial data by mukey. Shape: {mo_mu_agg_by_mukey.shape}")
        
        # Define columns needed for analysis/clustering based on original script
        # Adjust depth suffix based on ANALYSIS_DEPTH
        depth_suffix = f'_{ANALYSIS_DEPTH}cm'
        required_cols_analysis = [
            'mukey', 'muname', 'mjcmpnms', 'area_ac', 'complex', 'associan',
            'MnRs_dep', # Min restriction depth
            f'clay{depth_suffix}', f'sand{depth_suffix}', f'om{depth_suffix}', f'cec{depth_suffix}', f'bd{depth_suffix}',
            f'ec{depth_suffix}', f'pH{depth_suffix}', f'lep{depth_suffix}', f'ksat{depth_suffix}',
            f'awc{depth_suffix}', f'sar{depth_suffix}', # Added sar based on clustering columns
            # QC columns
            'compct_om', 'compct_cec', 'compct_ksat', 'compct_awc', 'compct_clay',
            'compct_bd', 'compct_pH', 'compct_lep', 'dmcmp_pct'
        ]
        # Filter required_cols_analysis to only include columns present in mo_mu_agg_by_mukey
        required_cols_analysis = [col for col in required_cols_analysis if col in mo_mu_agg_by_mukey.columns]
        analysis_dataset = mo_mu_agg_by_mukey[required_cols_analysis].copy()
        print(f"Created analysis dataset with columns: {analysis_dataset.columns.tolist()}")

        # Perform final QC checks (NA counts, data coverage) based on original script logic
        property_cols = [f'clay{depth_suffix}',f'sand{depth_suffix}', f'om{depth_suffix}', f'cec{depth_suffix}', f'bd{depth_suffix}',
                        f'ec{depth_suffix}', f'pH{depth_suffix}', f'lep{depth_suffix}', f'ksat{depth_suffix}',
                        f'awc{depth_suffix}', 'MnRs_dep'] # list of core property columns
        # Filter to existing columns
        property_cols = [col for col in property_cols if col in analysis_dataset.columns]
        
        analysis_dataset['count_NAs'] = analysis_dataset[property_cols].isna().sum(axis=1)
        print("NA counts per map unit (based on core properties):")
        print(analysis_dataset.groupby('count_NAs')['area_ac'].sum())

        # Example: Impute EC where it's the only NA (as per original script)
        ec_col = f'ec{depth_suffix}'
        if ec_col in analysis_dataset.columns:
            ec_na_mask = (analysis_dataset[ec_col].isna()) & (analysis_dataset['count_NAs'] == 1)
            if ec_na_mask.any():
                print(f"Imputing 0 for {ec_na_mask.sum()} instances where {ec_col} is the only NA.")
                analysis_dataset.loc[ec_na_mask, ec_col] = 0
                # Recalculate NA count
                analysis_dataset['count_NAs'] = analysis_dataset[property_cols].isna().sum(axis=1)
                print("Recalculated NA counts after EC imputation:")
                print(analysis_dataset.groupby('count_NAs')['area_ac'].sum())

        # Define minimum data coverage requirement (e.g., using compct_* columns)
        qc_cols = ['compct_om', 'compct_cec', 'compct_ksat', 'compct_awc', 'compct_clay',
                'compct_bd', 'compct_pH', 'compct_lep']
        qc_cols = [col for col in qc_cols if col in analysis_dataset.columns] # Filter existing
        
        if qc_cols:
            analysis_dataset["min_dat_cov"] = analysis_dataset[qc_cols].min(axis=1, skipna=True) # Use skipna=True
            # Apply quality filter (e.g., keep if min coverage >= 80% OR min coverage >= dominant component pct, AND no NAs in properties)
            qc_filter = (
                ((analysis_dataset["min_dat_cov"] >= 80) | (analysis_dataset["min_dat_cov"] >= analysis_dataset["dmcmp_pct"])) &
                (analysis_dataset["count_NAs"] == 0)
            )
            analysis_dataset_final = analysis_dataset[qc_filter].copy()
            print(f"Applied QC filter. Final dataset shape for clustering: {analysis_dataset_final.shape}")
            print(f"Final acreage sum: {analysis_dataset_final['area_ac'].sum():.2f}")
        else:
            print("Skipping QC filtering based on data coverage as QC columns are missing.")
            # If QC cannot be applied, decide whether to proceed with NAs or only count_NAs==0
            analysis_dataset_final = analysis_dataset[analysis_dataset['count_NAs'] == 0].copy()
            print(f"Proceeding with data where count_NAs == 0. Shape: {analysis_dataset_final.shape}")
            
        if analysis_dataset_final.empty:
            print("Error: No data remaining after final QC. Cannot proceed to clustering.")
            return
            
        # Save the final dataset used for clustering
        final_data_csv_path = os.path.join(OUTPUT_DIR, f'MO_{ANALYSIS_DEPTH}cm_for_clustering.csv')
        analysis_dataset_final.to_csv(final_data_csv_path, index=False)
        print(f"Saved final pre-clustering data to {final_data_csv_path}")
        analysis_dataset_final["mukey"].to_csv(os.path.join(OUTPUT_DIR, f'MO_{ANALYSIS_DEPTH}cm_mukey_clustering.csv'), index=False)
        # Also save a Parquet copy for downstream steps (clustering/VAE, etc.)
        try:
            final_df = analysis_dataset_final  # alias to match your requested variable name
            final_parquet = os.path.join(OUTPUT_DIR, "prepared_df.parquet")
            final_df.to_parquet(final_parquet, index=False)  # requires pyarrow or fastparquet
            print(f"Saved Parquet at: {final_parquet}")
        except Exception as e:
            print(f"Warning: could not write Parquet file ('prepared_df.parquet'): {e}")
    # --- Exception handling and finally block for overall workflow ---
    except Exception as e:
        log.critical(f"An unhandled exception occurred in the main workflow: {e}", exc_info=True)
    finally:
        # Calculate total duration
        overall_end_time = time.perf_counter()
        if 'overall_start_time' in locals():
             overall_duration = overall_end_time - overall_start_time
             log.info(f"--- Total Workflow Duration: {overall_duration:.2f} seconds ---")
        log.info("--- End of main workflow execution ---")

# # ---------------- CLI + run ----------------

def build_argparser():
    """
    Argument builder for running the SSURGO pipeline from CLI.
    NOTE: This function is provided without wiring it into main() so it won't
    change your current behavior unless you explicitly choose to use it.
    """
    p = argparse.ArgumentParser(description="SSURGO end-to-end processing for Missouri.")
    p.add_argument(
        "--base-dir",
        required=True,
        help=("Base data directory (contains mupoly.shp, mapunit.csv, component.csv, "
              "muagg.csv, corestrictions.csv, chorizon.csv, cfrag.csv)")
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write outputs (e.g., .../aggResult)"
    )
    p.add_argument(
        "--target-crs",
        required=True,
        default=None,
        help="Target CRS for spatial processing (default: EPSG:5070)."
    )
    p.add_argument(
        "--analysis-depth",
        type=int,
        choices=[10, 30, 100],
        required=True,
        help="Depth slice to aggregate horizon data over (default: 30)."
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity with -v, -vv, etc."
    )
    return p
def main(argv=None):
    # ✅ parse CLI args here
    args = build_argparser().parse_args(argv)

    BASE_DIR       = args.base_dir
    OUTPUT_DIR     = args.output_dir
    TARGET_CRS     = args.target_crs
    ANALYSIS_DEPTH = args.analysis_depth

    # --- now run your workflow, replacing DEFAULT_* with these ---
    print(f"Running pipeline with:")
    print(f"  Base dir       : {BASE_DIR}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print(f"  Target CRS     : {TARGET_CRS}")
    print(f"  Analysis depth : {ANALYSIS_DEPTH}")

    # 🔽 call your workflow code here (instead of the old main body)
    # e.g. move your whole existing workflow logic into another function
    return 0

# Runbook compatibility
def run_main(argv=None) -> int:
    return main(argv)

if __name__ == "__main__":
    sys.exit(main())

