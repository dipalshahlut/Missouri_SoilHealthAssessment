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

def load_horizon_data(horizon_csv: str, cfrag_csv: str) -> pd.DataFrame:
    """
    Load horizon (chorizon.csv) and coarse fragment (cfrag.csv) tables,
    normalize column names, attach fragvol_r, ensure awc_r exists (warn if missing),
    coerce key numeric columns, and return a trimmed dataframe.

    Returns
    -------
    pd.DataFrame
        Columns (when present): [
            'cokey','chkey','hzdept_r','hzdepb_r','fragvol_r',
            'claytotal_r','silttotal_r','sandtotal_r','om_r','cec7_r',
            'dbthirdbar_r','kwfact','ec_r','ph1to1h2o_r','sar_r',
            'caco3_r','gypsum_r','lep_r','ksat_r','awc_r'
        ]
    """
    if not os.path.exists(horizon_csv):
        raise FileNotFoundError(f"Missing horizon file: {horizon_csv}")
    if not os.path.exists(cfrag_csv):
        raise FileNotFoundError(f"Missing coarse fragment file: {cfrag_csv}")

    hz = pd.read_csv(horizon_csv, low_memory=False)
    cf = pd.read_csv(cfrag_csv, low_memory=False)

    # Normalize column names to lowercase
    for df in (hz, cf):
        df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    # Ensure keys exist
    if "cokey" not in hz.columns:
        raise KeyError("Horizon table must include 'cokey'.")
    if "hzdept_r" not in hz.columns or "hzdepb_r" not in hz.columns:
        raise KeyError("Horizon table must include 'hzdept_r' and 'hzdepb_r'.")

    # --- Ensure awc_r exists (warn + fill with NaN if missing) ---
    if "awc_r" not in hz.columns:
        log.warning(
            "Column 'awc_r' not found in horizon data. "
            "Filling with NaN — AWC values will be missing downstream."
        )
        hz["awc_r"] = np.nan

    # --- Attach fragvol_r from cfrag ---
    frag_added = False
    if not cf.empty:
        # Prefer joining by chkey if available in both
        if "chkey" in hz.columns and "chkey" in cf.columns:
            frag = (
                cf.groupby("chkey", as_index=False)["fragvol_r"]
                  .mean()  # average across fragment records within a horizon
            )
            hz = hz.merge(frag, on="chkey", how="left")
            frag_added = True
        else:
            # Fallback: average by (cokey, hzdept_r, hzdepb_r) if present in cfrag
            join_cols = [c for c in ["cokey", "hzdept_r", "hzdepb_r"] if c in cf.columns]
            if len(join_cols) >= 2 and "fragvol_r" in cf.columns:
                frag = (
                    cf.groupby(join_cols, as_index=False)["fragvol_r"]
                      .mean()
                )
                hz = hz.merge(frag, on=join_cols, how="left")
                frag_added = True

    if not frag_added and "fragvol_r" not in hz.columns:
        # If still missing, add as NaN (some datasets don’t have cfrag)
        log.info("No coarse fragment join performed; setting 'fragvol_r' to NaN.")
        hz["fragvol_r"] = np.nan

    # --- Coerce numerics for key columns (ignore if absent) ---
    numeric_cols = [
        "hzdept_r", "hzdepb_r", "fragvol_r", "dbthirdbar_r", "awc_r",
        "claytotal_r", "silttotal_r", "sandtotal_r", "om_r",
        "cec7_r", "kwfact", "ec_r", "ph1to1h2o_r", "sar_r",
        "caco3_r", "gypsum_r", "lep_r", "ksat_r",
    ]
    for col in numeric_cols:
        if col in hz.columns:
            hz[col] = pd.to_numeric(hz[col], errors="coerce")

    # --- Keep only the columns we actually use downstream (plus keys) ---
    keep_core = {
        "cokey", "chkey", "hzdept_r", "hzdepb_r", "fragvol_r",
        "claytotal_r", "silttotal_r", "sandtotal_r", "om_r", "cec7_r",
        "dbthirdbar_r", "kwfact", "ec_r", "ph1to1h2o_r", "sar_r",
        "caco3_r", "gypsum_r", "lep_r", "ksat_r", "awc_r",
    }
    keep = [c for c in hz.columns if c in keep_core]

    # Final sanity: drop obviously bad depth rows
    out = hz[keep].copy()
    out = out[out["hzdepb_r"] > out["hzdept_r"]]

    return out



# ---------------------------------------------------------------------
# 2) Prepare / Clean
# ---------------------------------------------------------------------

def prepare_horizon_data(
    horizon_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    reskinds_by_cokey: Optional[pd.DataFrame] = None,
    rock_na_to_0: bool = True,
) -> pd.DataFrame:
    """
    Filter horizons to the study area (via comp_df) and attach component metadata.

    Parameters
    ----------
    horizon_df : pd.DataFrame
        Output of `load_horizon_data`.
    comp_df : pd.DataFrame
        Component metadata already filtered to study area (e.g., Missouri).
        Must include: ['cokey','mukey','compname','comppct_r','majcompflag'].
    reskinds_by_cokey : pd.DataFrame or None
        Optional table with ['cokey','reskinds'] summarizing restriction kinds.
    rock_na_to_0 : bool
        If True, set missing fragvol_r to 0 (common pragmatic choice).

    Returns
    -------
    pd.DataFrame
        Horizon rows limited to **major components** in the study area,
        with component attributes attached and cleaned numeric fields.
    """
    required_cols = {"cokey", "mukey", "compname", "comppct_r", "majcompflag"}
    missing = required_cols - set(comp_df.columns)
    if missing:
        raise KeyError(f"comp_df missing required columns: {sorted(missing)}")

    # Keep only components in the study area & mark majors
    comp = comp_df.copy()
    comp["mukey"] = comp["mukey"].astype(str)
    comp["majcompflag"] = comp["majcompflag"].astype(str).str.strip()

    majors = comp[comp["majcompflag"].str.upper().str.startswith("YES")].copy()
    if majors.empty:
        log.warning("No 'major' components found; using all components instead.")
        majors = comp.copy()

    # Attach component metadata to horizons
    cols_to_keep = ["cokey", "mukey", "compname", "comppct_r"]
    hz = horizon_df.merge(majors[cols_to_keep], on="cokey", how="inner")

    # Optionally attach restriction kinds per component
    if reskinds_by_cokey is not None and not reskinds_by_cokey.empty:
        if {"cokey", "reskinds"} <= set(reskinds_by_cokey.columns):
            hz = hz.merge(reskinds_by_cokey[["cokey", "reskinds"]], on="cokey", how="left")
        else:
            log.warning("reskinds_by_cokey missing expected columns; skipping attach.")

    # Cleaning tweaks
    if rock_na_to_0 and "fragvol_r" in hz.columns:
        hz["fragvol_r"] = hz["fragvol_r"].fillna(0.0)

    # Drop obviously bad depth rows
    hz = hz[hz["hzdepb_r"] > hz["hzdept_r"]].copy()

    # Ensure types
    for c in ["mukey"]:
        if c in hz.columns:
            hz[c] = hz[c].astype(str)

    return hz


# ---------------------------------------------------------------------
# 3) Aggregate horizon ➜ component at a depth slice
# ---------------------------------------------------------------------

def aggregate_horizons_depth_slice(
    hz_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    depth_cm: int,
    out_dir: str,
    tag: str,
) -> pd.DataFrame:
    """
    Aggregate horizon variables to the component level up to `depth_cm` and save to CSV.

    Parameters
    ----------
    hz_df : pd.DataFrame
        Prepared horizon rows (from `prepare_horizon_data`).
    comp_df : pd.DataFrame
        Component metadata (at least ['cokey','mukey','compname','comppct_r']).
    depth_cm : int
        Depth threshold in centimeters (e.g., 10, 30, 100).
    out_dir : str
        Output directory; CSV will be written here.
    tag : str
        A short label used in the filename (e.g., "30cm").

    Returns
    -------
    pd.DataFrame
        Component-level aggregates with columns like clay_30cm, om_30cm, kgOrg.m2_30cm, awc_30cm, etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    comp_agg = horizon_to_comp(
        horizon_df=hz_df,
        depth=depth_cm,
        comp_df=comp_df,
        vars_of_interest=None,  # use defaults inside horizon_to_comp
        varnames=None
    )

    # Save
    out_csv = os.path.join(out_dir, f"comp_mo_{tag}.csv")
    comp_agg.to_csv(out_csv, index=False)
    log.info("Wrote component aggregates for %s: %s", tag, out_csv)
    return comp_agg


# ---------------------------------------------------------------------
# 4) QC summaries
# ---------------------------------------------------------------------

def quality_check_aggregation(comp_agg_df: pd.DataFrame, depth_cm: int) -> Dict[str, pd.DataFrame]:
    """
    Quick QC coverage summaries for component aggregates at the given depth.

    For each metric column "*_{depth}cm":
        - Compute, per MU, the total component percent (comppct) that *has data* for that metric.
        - The result is an approximate coverage indicator in [% of component composition].

    Parameters
    ----------
    comp_agg_df : pd.DataFrame
        Output of `aggregate_horizons_depth_slice`: must include
        ['mukey','cokey','compname','comppct', <metric_*_{depth}cm>...].
    depth_cm : int
        Depth used to derive the metric columns (e.g., 30).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping metric_name -> DataFrame with columns ['mukey','compct_with_data'] (0..100).
    """
    if comp_agg_df.empty:
        return {}

    req_cols = {"mukey", "comppct"}
    if not req_cols.issubset(comp_agg_df.columns):
        missing = req_cols - set(comp_agg_df.columns)
        raise KeyError(f"comp_agg_df missing required columns: {sorted(missing)}")

    df = comp_agg_df.copy()
    df["mukey"] = df["mukey"].astype(str)

    # Identify metric columns for this depth
    suffix = f"_{depth_cm}cm"
    metric_cols = [c for c in df.columns if c.endswith(suffix) and not c.startswith(("kgOrg.m2", "awc_"))]
    # Include kgOrg and awc explicitly as they use slightly different base names
    if f"kgOrg.m2_{depth_cm}cm" in df.columns:
        metric_cols.append(f"kgOrg.m2_{depth_cm}cm")
    if f"awc_{depth_cm}cm" in df.columns:
        metric_cols.append(f"awc_{depth_cm}cm")

    out: Dict[str, pd.DataFrame] = {}

    for col in metric_cols:
        mask_has_data = df[col].notna()
        # Sum component % where this metric is present
        compct = (df.loc[mask_has_data]
                    .groupby("mukey")["comppct"]
                    .sum()
                    .reset_index(name="compct_with_data"))
        # Normalize to 0..100 (component percents typically sum to ~100 per MU)
        # If your comppct is already 0..100, this is already in %
        # If your comppct is 0..1, scale accordingly here.
        out[col] = compct.sort_values("compct_with_data", ascending=False)

    return out
