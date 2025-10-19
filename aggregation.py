
#!/usr/bin/env python3
"""
aggregation.py — repo-aligned pipeline with a small CLI wrapper.

Replicates the uploaded aggregation.py behavior:
  1) Load MU polygons, reproject to EPSG:5070, compute area, MUKEY→mukey
  2) Load mapunit/component/muagg, right-join on mukey, filter to spatial MUKEYs
  3) Component summaries and flags
  4) Restrictions: depths, summaries, percentages
  5) Horizons: load/prepare, aggregate 10/30/100 cm, QC
  6) steps 3-5 Aggregates components, restrictions, and horizons
  7) Produces spatial outputs and an aggregated dataframe (Parquet)
  
Outputs (written to OUTPUT_DIR):
  - prepared_df.parquet
  
Usage:
python aggregation.py \
  --base-dir /path/to/data \
  --output-dir /path/to/data/aggResult \
  --target-crs EPSG:5070 \
  --analysis-depth 30

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os, time, logging, argparse, sys,re
from typing import Dict, Tuple

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
# ---------------- helpers ----------------
def _configure_logging(verbosity: int) -> None:
    # verbosity: 0->WARNING, 1->INFO, >=2->DEBUG
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
# ---------------- main pipeline ----------------
def run_pipeline(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SSURGO end-to-end processing for Missouri.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help=("Base data directory (contains mupoly.shp, mapunit.csv, component.csv, "
              "muagg.csv, corestrictions.csv, chorizon.csv, cfrag.csv)")
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write outputs (e.g., .../aggResult)"
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:5070",
        help="Target CRS for spatial processing (default: EPSG:5070)."
    )
    parser.add_argument(
        "--analysis-depth",
        type=int,
        choices=[10, 30, 100],
        default=30,
        help="Depth slice to aggregate horizon data over (default: 30)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity with -v, -vv, etc."
    )
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    BASE_DIR       = args.base_dir
    OUTPUT_DIR     = args.output_dir
    TARGET_CRS     = args.target_crs
    ANALYSIS_DEPTH = args.analysis_depth

    MU_COUNTY_SHP          = os.path.join(BASE_DIR, "MO_County_Boundaries.shp")  # unused here, kept for compatibility
    MU_POLY_PATH           = os.path.join(BASE_DIR, "mupoly.shp")
    MAPUNIT_CSV_PATH       = os.path.join(BASE_DIR, "mapunit.csv")
    COMPONENT_CSV_PATH     = os.path.join(BASE_DIR, "component.csv")
    MUAGG_CSV_PATH         = os.path.join(BASE_DIR, "muagg.csv")
    RESTRICTIONS_CSV_PATH  = os.path.join(BASE_DIR, "corestrictions.csv")
    HORIZON_CSV_PATH       = os.path.join(BASE_DIR, "chorizon.csv")
    CFRAG_CSV_PATH         = os.path.join(BASE_DIR, "cfrag.csv")

    log.info("Starting main workflow...")
    overall_start_time = time.perf_counter()
    print("Starting SSURGO Data Processing Workflow...")

    try:
        # --- 1. Load and Prepare Initial Data ---
        log.info("--- Workflow Step: Geographic Processing ---")
        print("\\n--- Step 1: Loading Initial Data ---")

        mo_mu_shp_initial = load_and_filter_spatial_data_new(MU_POLY_PATH)
        mo_mu_shp = reproject_and_calculate_area(mo_mu_shp_initial, TARGET_CRS)
        print(f"Reproject MU data for MO: {mo_mu_shp.shape}")

        mu_data  = pd.read_csv(MAPUNIT_CSV_PATH, na_values=['', ' '])
        mu_data['mukey'] = mu_data['mukey'].astype(str)

        comp_   = pd.read_csv(COMPONENT_CSV_PATH, na_values=['', ' '])
        map_df  = pd.read_csv(MUAGG_CSV_PATH, na_values=['', ' '])  # muaggatt data

        comp_df = pd.merge(comp_, map_df, on='mukey', how='right')  # repo-aligned behavior
        comp_df['mukey'] = comp_df['mukey'].astype(str)

        mukeys_in_spatial = mo_mu_shp['mukey'].astype(str).unique()
        print(f"Unique MUKEY count for MO: {len(mukeys_in_spatial)}")
        mu_data_mo  = mu_data[mu_data['mukey'].isin(mukeys_in_spatial)].copy()
        print(f"Filtered Map Unit Data for MO: {mu_data_mo.shape}")
        comp_data_mo = comp_df[comp_df['mukey'].isin(mukeys_in_spatial)].copy()
        print(f"Filtered component data for MO: {comp_data_mo.shape}")

        # --- 2. Process Component Data ---
        log.info("--- Workflow Step: Component data Processing ---")
        print("\\n--- Step 2: Processing Component Data ---")

    # Normalize majcompflag values to avoid trailing space issues and derive a clean boolean
        if 'majcompflag' in comp_data_mo.columns:
            comp_data_mo['majcompflag'] = comp_data_mo['majcompflag'].astype(str).str.strip()
        if comp_data_mo['comppct_r'].isna().any():
            print("Warning: NAs found in 'comppct_r'. Dropping these rows.")
            comp_data_mo.dropna(subset=['comppct_r'], inplace=True)

        # Restore original major-component logic used elsewhere in the repo:
        comp_data_mo.loc[comp_data_mo['comppct_r'] >= 15, 'majcompflag'] = 'Yes'
        comp_data_mo.loc[comp_data_mo['comppct_r'] <  15, 'majcompflag'] = 'No'

        # Derive a reliable 'is_major' flag instead of mutating the source
        comp_data_mo['is_major'] = comp_data_mo['comppct_r'] >= 15

        # Summaries by mukey
        majcomps_no_by_mukey = (comp_data_mo[comp_data_mo['is_major']]
                                .groupby('mukey').size().reset_index(name='majcomp_no'))
        majcompnames_by_mukey = (comp_data_mo[comp_data_mo['is_major']]
                                .groupby('mukey')['compname'].apply(concat_names)
                                .reset_index().rename(columns={'compname': 'majcompnames'}))
        majcomp_taxorders_by_mukey = (comp_data_mo[comp_data_mo['is_major']]
                                    .groupby('mukey')['taxorder'].apply(concat_names)
                                    .reset_index().rename(columns={'taxorder': 'taxorders'}))
        compnames_by_mukey = (comp_data_mo.groupby('mukey')['compname']
                            .apply(concat_names).reset_index().rename(columns={'compname': 'compnames'}))
        domcomp_pct_by_mukey = (comp_data_mo.groupby('mukey')['comppct_r']
                                .max().reset_index().rename(columns={'comppct_r': 'domcomppct'}))
        majcomp_pct_by_mukey = (comp_data_mo[comp_data_mo['is_major']]
                                .groupby('mukey')['comppct_r'].sum()
                                .reset_index().rename(columns={'comppct_r': 'majcomppct'}))
        print("Generated component summaries by mukey.")

        # --- 3. Process Restrictions ---
        log.info("--- Workflow Step: Restriction Processing ---")
        print("\\n--- Step 3: Processing Restrictions ---")
        restrictions_mo = load_and_process_restrictions(RESTRICTIONS_CSV_PATH, comp_data_mo)

        restriction_depths_agg = {}
        restriction_pcts_agg   = {}
        if not restrictions_mo.empty:
            restriction_depths_agg = aggregate_restriction_depths(restrictions_mo)
            reskinds_by_cokey_summary, reskinds_by_mukey_summary = create_restriction_summary(
                restrictions_mo, comp_data_mo
            )
            restriction_pcts_agg = calculate_restriction_percentages(
                comp_data_mo, reskinds_by_cokey_summary
            )
        else:
            print("No MO restrictions found. Skipping restriction depth/percentage aggregation.")
            reskinds_by_cokey_summary = pd.DataFrame(columns=['cokey', 'reskinds'])
            reskinds_by_mukey_summary = pd.DataFrame(columns=['mukey', 'reskinds'])

        # --- 4. Process Horizons ---
        log.info("--- Workflow Step: Horizon Processing ---")
        print("\\n--- Step 4: Processing Horizons ---")
        horizon_df = load_horizon_data(HORIZON_CSV_PATH, CFRAG_CSV_PATH)
        horizons_mo_majcomps = prepare_horizon_data(
            horizon_df, comp_data_mo, reskinds_by_cokey_summary, ROCK_NA_TO_0
        )

        comp_mo_10cm  = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 10,  OUTPUT_DIR, "10cm")
        comp_mo_30cm  = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 30,  OUTPUT_DIR, "30cm")
        comp_mo_100cm = aggregate_horizons_depth_slice(horizons_mo_majcomps, comp_data_mo, 100, OUTPUT_DIR, "100cm")

        qc_results_10cm  = quality_check_aggregation(comp_mo_10cm,  10)
        qc_results_30cm  = quality_check_aggregation(comp_mo_30cm,  30)
        qc_results_100cm = quality_check_aggregation(comp_mo_100cm, 100)

        # --- 5. Integrate Data into Map Unit Polygons ---
        print("\\n--- Step 5: Integrating Data into Map Unit Polygons ---")

        mo_mu_shp['mukey'] = mo_mu_shp['mukey'].astype(str)
        print(f"The Mo_mu_shp shape is : {mo_mu_shp.shape}")

        # Convert keys to string in summary dfs
        for df_ in [majcomps_no_by_mukey, majcomp_taxorders_by_mukey, domcomp_pct_by_mukey,
                    majcomp_pct_by_mukey, majcompnames_by_mukey, compnames_by_mukey, mu_data_mo]:
            df_['mukey'] = df_['mukey'].astype(str)

        # Add component summaries to spatial data
        mo_mu_shp['mjcps_no']  = mo_mu_shp['mukey'].map(majcomps_no_by_mukey.set_index('mukey')['majcomp_no'])
        mo_mu_shp['txorders']  = mo_mu_shp['mukey'].map(majcomp_taxorders_by_mukey.set_index('mukey')['taxorders'])
        mo_mu_shp['dmcmp_pct'] = mo_mu_shp['mukey'].map(domcomp_pct_by_mukey.set_index('mukey')['domcomppct'])
        mo_mu_shp['mjcmp_pct'] = mo_mu_shp['mukey'].map(majcomp_pct_by_mukey.set_index('mukey')['majcomppct'])
        mo_mu_shp['mjcmpnms']  = mo_mu_shp['mukey'].map(majcompnames_by_mukey.set_index('mukey')['majcompnames'])
        mo_mu_shp['compnames'] = mo_mu_shp['mukey'].map(compnames_by_mukey.set_index('mukey')['compnames'])
        mo_mu_shp['muname']    = mo_mu_shp['mukey'].map(mu_data_mo.set_index('mukey')['muname'])
        mo_mu_shp['complex']   = mo_mu_shp['muname'].str.contains('complex', case=False, na=False).map({True: 'Yes', False: 'No'})
        mo_mu_shp['association'] = mo_mu_shp['muname'].str.contains('association', case=False, na=False).map({True: 'Yes', False: 'No'})

        # Add restriction summary and individual flags
        print("Adding restriction summaries and flags...")
        if not reskinds_by_mukey_summary.empty:
            reskinds_by_mukey_summary['mukey'] = reskinds_by_mukey_summary['mukey'].astype(str)
            mo_mu_shp['restrict'] = mo_mu_shp['mukey'].map(
                reskinds_by_mukey_summary.set_index('mukey')['reskinds']
            ).fillna('None')
            # flags
            src = mo_mu_shp['restrict'].fillna('')
            mo_mu_shp['Lithic']   = src.str.contains('Lithic bedrock', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Paralith'] = src.str.contains('Paralithic bedrock', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Fragipan'] = src.str.contains('Fragipan', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['ATC']      = src.str.contains('Abrupt textural change', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Natric']   = src.str.contains('Natric', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['SCTS']     = src.str.contains('Strongly contrasting textural stratification', na=False).map({True: 'Yes', False: 'No'})
            mo_mu_shp['Misc_Res'] = src.str.contains('Densic material|Cemented horizon|Petrocalcic|Undefined', na=False).map({True: 'Yes', False: 'No'})
        else:
            mo_mu_shp['restrict'] = 'None'
            for flag in ['Lithic', 'Paralith', 'Fragipan', 'ATC', 'Natric', 'SCTS', 'Misc_Res']:
                mo_mu_shp[flag] = 'No'

        # Rock Outcrop flag by component name + component percent (% of MU area by comps with 'Rock outcrop')
        mo_mu_shp['Rock_OC'] = mo_mu_shp['compnames'].str.contains('Rock outcrop', na=False).map({True: 'Yes', False: 'No'})
        rock_pct_by_mukey = (
            comp_data_mo[comp_data_mo['compname'].str.contains('Rock outcrop', na=False)]
            .groupby('mukey', as_index=False)['comppct_r']
            .sum()
            .rename(columns={'comppct_r': 'compct_sum'})
        )
        rock_pct_by_mukey['mukey'] = rock_pct_by_mukey['mukey'].astype(str)

        print(f"MU polygon GeoDataFrame shape: {mo_mu_shp.shape}")

        # Add restriction depths
        print("Adding restriction depths...")
        depth_mapping = {
            'Lithic':   ('Lthc_dep', 'Lithic bedrock'),
            'Paralith': ('Plth_dep', 'Paralithic bedrock'),
            'Fragipan': ('Frpn_dep', 'Fragipan'),
            'ATC':      ('ATC_dep',  'Abrupt textural change'),
            'Natric':   ('Natr_dep', 'Natric'),
            'SCTS':     ('SCTS_dep', 'Strongly contrasting textural stratification'),
            'Misc_Res': ('MRes_dep', 'Miscellaneous'),
        }
        depth_cols_added = []
        for flag_col, (depth_col, res_key) in depth_mapping.items():
            mo_mu_shp[depth_col] = ASSUMED_DEPTH
            if res_key in restriction_depths_agg and not restriction_depths_agg[res_key].empty:
                depth_df = restriction_depths_agg[res_key].copy()
                depth_df['mukey'] = depth_df['mukey'].astype(str)
                idx = depth_df.set_index('mukey')
                def _pick_depth(row):
                    if row.get(flag_col) == 'Yes' and row['mukey'] in idx.index:
                        return idx.loc[row['mukey']]['resdept_r']
                    return row[depth_col]
                mo_mu_shp[depth_col] = mo_mu_shp.apply(_pick_depth, axis=1)
                depth_cols_added.append(depth_col)
            else:
                log.debug(f"No aggregated depth data found for {res_key}.")
        if depth_cols_added:
            mo_mu_shp['MnRs_dep'] = mo_mu_shp[depth_cols_added].min(axis=1)
            print("Calculated minimum restriction depth.")
        else:
            print("Minimum restriction depth calculation skipped as no depth columns were added.")
            mo_mu_shp['MnRs_dep'] = ASSUMED_DEPTH

        # Add restriction percentages
        print("Adding restriction percentages...")
        pct_mapping = {
            'Lithic':   ('Lthc_pct', 'Lithic bedrock'),
            'Paralith': ('Plth_pct', 'Paralithic bedrock'),
            'Fragipan': ('Frpn_pct', 'Fragipan'),
            'ATC':      ('ATC_pct',  'Abrupt textural change'),
            'Natric':   ('Natr_pct', 'Natric'),
            'SCTS':     ('SCTS_pct', 'Strongly contrasting textural stratification'),
            'Misc_Res': ('MRes_pct', 'Miscellaneous'),
        }
        pct_cols_added = []
        for flag_col, (pct_col, res_key) in pct_mapping.items():
            mo_mu_shp[pct_col] = 0.0
            if res_key in restriction_pcts_agg and not restriction_pcts_agg[res_key].empty:
                pct_df = restriction_pcts_agg[res_key].copy()
                pct_df['mukey'] = pct_df['mukey'].astype(str)
                idx = pct_df.set_index('mukey')
                def _pick_pct(row):
                    if row.get(flag_col) == 'Yes' and row['mukey'] in idx.index:
                        return float(idx.loc[row['mukey']]['compct_sum'])
                    return row[pct_col]
                mo_mu_shp[pct_col] = mo_mu_shp.apply(_pick_pct, axis=1)
                pct_cols_added.append(pct_col)
            else:
                log.debug(f"No aggregated percentage data found for {res_key}.")
        # Rock outcrop % from components by name
        mo_mu_shp['RckOC_pct'] = mo_mu_shp['mukey'].map(rock_pct_by_mukey.set_index('mukey')['compct_sum']).fillna(0.0)

        # Show stats if any pct was added
        if pct_cols_added or 'RckOC_pct' in mo_mu_shp.columns:
            cols = pct_cols_added + (['RckOC_pct'] if 'RckOC_pct' in mo_mu_shp.columns else [])
            print(mo_mu_shp[cols].describe())

        # Add aggregated horizon data (for ANALYSIS_DEPTH cm)
        print(f"Adding aggregated horizon data for {ANALYSIS_DEPTH}cm...")
        comp_agg_dict = {10: comp_mo_10cm, 30: comp_mo_30cm, 100: comp_mo_100cm}
        comp_analysis_agg = comp_agg_dict.get(ANALYSIS_DEPTH)

        if comp_analysis_agg is not None and not comp_analysis_agg.empty:
            # Exclude non-numeric/ID columns
            cols_to_exclude = ['cokey', 'mukey', 'compname', 'comppct'] + [c for c in comp_analysis_agg.columns if c.endswith('_zero')]
            varnames_agg = [c for c in comp_analysis_agg.columns if c not in cols_to_exclude]

            print(f"Aggregating variables to mapunit level: {len(varnames_agg)} variables.")
            mu_analysis_agg = MUAggregate_wrapper(df=comp_analysis_agg, varnames=varnames_agg)

            if mu_analysis_agg is not None and not mu_analysis_agg.empty:
                mu_analysis_agg['mukey'] = mu_analysis_agg['mukey'].astype(str)
                mo_mu_shp = mo_mu_shp.merge(mu_analysis_agg, on='mukey', how='left')
                print(f"Merged {ANALYSIS_DEPTH}cm aggregated horizon data. New shape: {mo_mu_shp.shape}")
                if varnames_agg:
                    first_col = varnames_agg[0]
                    print(f"NA count for first merged column ({first_col}): {mo_mu_shp[first_col].isna().sum()}")
            else:
                print(f"Mapunit aggregation failed for {ANALYSIS_DEPTH}cm data (empty result).")
        else:
            print(f"Skipping merge of {ANALYSIS_DEPTH}cm horizon data as it's empty/None.")

        # Add QC metrics (component percentage coverage for key variables)
        print("Adding QC metrics (component coverage)...")
        qc_results_dict = {10: qc_results_10cm, 30: qc_results_30cm, 100: qc_results_100cm}
        qc_analysis = qc_results_dict.get(ANALYSIS_DEPTH)
        if qc_analysis:
            qc_vars_mapping = {
                f'om_{ANALYSIS_DEPTH}cm':  'compct_om',
                f'cec_{ANALYSIS_DEPTH}cm': 'compct_cec',
                f'ksat_{ANALYSIS_DEPTH}cm':'compct_ksat',
                f'awc_{ANALYSIS_DEPTH}cm': 'compct_awc',
                f'clay_{ANALYSIS_DEPTH}cm':'compct_clay',
                f'bd_{ANALYSIS_DEPTH}cm':  'compct_bd',
                f'ec_{ANALYSIS_DEPTH}cm':  'compct_ec',
                f'pH_{ANALYSIS_DEPTH}cm':  'compct_pH',
                f'lep_{ANALYSIS_DEPTH}cm': 'compct_lep',
            }
            for qc_var, target_col in qc_vars_mapping.items():
                qcdf = qc_analysis.get(qc_var)
                if qcdf is not None and not qcdf.empty:
                    qcdf = qcdf.copy()
                    qcdf['mukey'] = qcdf['mukey'].astype(str)
                    mo_mu_shp[target_col] = mo_mu_shp['mukey'].map(qcdf.set_index('mukey')['comppct_tot'])
                    print(f"Added QC column: {target_col}")
                else:
                    print(f"Skipping QC column {target_col} as source data was missing.")
                    mo_mu_shp[target_col] = np.nan
        else:
            print(f"Skipping addition of QC metrics as QC results for depth {ANALYSIS_DEPTH}cm are missing.")

        print(f"The MU polygon file shape before clustering is: {mo_mu_shp.shape}")

        # --- 6. Final Preparation and QC before Clustering ---
        log.info("--- Workflow Step: Final data Processing ---")
        print("\\n--- Step 6: Final Preparation and QC ---")

        acres_by_mukey = mo_mu_shp.groupby('mukey', as_index=False)['area_ac'].sum()
        mu_one = mo_mu_shp.drop_duplicates(subset='mukey', keep='first').copy()
        mu_one.drop(columns=['area_ac'], inplace=True, errors='ignore')
        mu_one = mu_one.merge(acres_by_mukey, on='mukey', how='left')

        # Select analysis columns & apply QC
        depth_suffix = f'_{ANALYSIS_DEPTH}cm'
        required_cols = [c for c in [
            'mukey','muname','mjcmpnms','area_ac','complex','association','MnRs_dep',
            f'clay{depth_suffix}', f'sand{depth_suffix}', f'om{depth_suffix}', f'cec{depth_suffix}',
            f'bd{depth_suffix}', f'ec{depth_suffix}', f'pH{depth_suffix}', f'lep{depth_suffix}',
            f'ksat{depth_suffix}', f'awc{depth_suffix}', f'sar{depth_suffix}',
            'compct_om','compct_cec','compct_ksat','compct_awc','compct_clay',
            'compct_bd','compct_pH','compct_lep','dmcmp_pct'
        ] if c in mu_one.columns]
        analysis_dataset = mu_one[required_cols].copy()

        prop_cols = [c for c in [
            f'clay{depth_suffix}', f'sand{depth_suffix}', f'om{depth_suffix}', f'cec{depth_suffix}',
            f'bd{depth_suffix}', f'ec{depth_suffix}', f'pH{depth_suffix}', f'lep{depth_suffix}',
            f'ksat{depth_suffix}', f'awc{depth_suffix}', 'MnRs_dep'
        ] if c in analysis_dataset.columns]

        analysis_dataset['count_NAs'] = analysis_dataset[prop_cols].isna().sum(axis=1)

        ec_col = f'ec{depth_suffix}'
        if ec_col in analysis_dataset.columns:
            only_ec_na = analysis_dataset[ec_col].isna() & (analysis_dataset['count_NAs'] == 1)
            if only_ec_na.any():
                analysis_dataset.loc[only_ec_na, ec_col] = 0
                analysis_dataset['count_NAs'] = analysis_dataset[prop_cols].isna().sum(axis=1)

        qc_cols = [c for c in ['compct_om','compct_cec','compct_ksat','compct_awc',
                               'compct_clay','compct_bd','compct_pH','compct_lep']
                   if c in analysis_dataset.columns]
        if qc_cols:
            analysis_dataset['min_dat_cov'] = analysis_dataset[qc_cols].min(axis=1, skipna=True)
            qc_mask = (((analysis_dataset['min_dat_cov'] >= 80) |
                        (analysis_dataset['min_dat_cov'] >= analysis_dataset['dmcmp_pct'])) &
                       (analysis_dataset['count_NAs'] == 0))
            analysis_dataset_final = analysis_dataset[qc_mask].copy()
        else:
            analysis_dataset_final = analysis_dataset[analysis_dataset['count_NAs'] == 0].copy()

        # 3) WRITE the exact files Stage-2 expects
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        prepared = analysis_dataset_final.copy()

        print("prepared_df rows:", len(prepared), "cols:", len(prepared.columns))
        if "mukey" not in prepared.columns:
            raise RuntimeError("analysis_dataset_final must contain 'mukey' before saving prepared_df.parquet")
        prepared_path = out_dir / "prepared_df.parquet"
        prepared.to_parquet(prepared_path, index=False)
        print(f"Saved Parquet for Stage-2: {prepared_path}")
        # persist MUKEY order for Stage-2 reindexing
        np.save(out_dir / "prepared_row_keys.npy", prepared["mukey"].astype(str).to_numpy())
        print("Saved prepared_row_keys.npy")

        return 0

    except Exception as e:
        log.critical(f"An unhandled exception occurred in the main workflow: {e}", exc_info=True)
        return 2
    finally:
        overall_end_time = time.perf_counter()
        duration = overall_end_time - overall_start_time
        log.info(f"--- Total Workflow Duration: {duration:.2f} seconds ---")
        log.info("--- End of main workflow execution ---")

# Thin main wrapper (keeps CLI stable)
def main(argv=None) -> int:
    return run_pipeline(argv)

# Runbook compatibility
def run_main(argv=None) -> int:
    return run_pipeline(argv)

if __name__ == "__main__":
    sys.exit(main())
