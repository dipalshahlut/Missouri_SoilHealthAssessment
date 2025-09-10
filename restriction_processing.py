#!/usr/bin/env python3
# restriction_processing.py
"""
Restriction processing utilities for SSURGO-style datasets.

Functions
---------
load_and_process_restrictions(restrictions_path, comp_data_mo)
    Load corestrictions, filter to study-area components, attach component info.

aggregate_restriction_depths(restrictions_mo)
    Aggregate restriction depths (top/bottom) per mukey by restriction kind.

create_restriction_summary(restrictions_mo, comp_data_mo)
    Build summaries of restriction kinds: per component (cokey) and per map unit (mukey).

calculate_restriction_percentages(comp_data_mo, reskinds_by_cokey)
    For each restriction kind of interest, compute the total component percentage per mukey.

Notes
-----
- This module assumes `comp_data_mo` has already been filtered to your study area (e.g., Missouri).
- Expected key columns:
    * Restrictions table: ['cokey','reskind','resdept_r','resdepb_r'] (names may vary; we normalize commonly used variants)
    * Components:          ['cokey','mukey','comppct_r','majcompflag','compname', ...]
- All joins are carried out on 'cokey' for component-level alignment and 'mukey' for MU-level outputs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Common aliases we normalize to expected names
_RESTRICTION_ALIASES = {
    "reskind": {"reskind", "res_kind", "restriction", "restrictionkind"},
    "resdept_r": {"resdept_r", "restrictiodepth_r", "resdeptmin_r", "restdept_r"},
    "resdepb_r": {"resdepb_r", "restrictiondepth_r", "resdeptmax_r", "restdepb_r"},
    "cokey": {"cokey", "co_key"},
}

# A small canonical set you may want to track explicitly
_DEFAULT_KINDS_OF_INTEREST: List[str] = [
    "Lithic bedrock",
    "Paralithic bedrock",
    "Fragipan",
    "Natric",
    "Duripan",
    "Cemented horizon",
    "Densic contact",
    "Strongly contrasting textural stratification",
    "Adverse texture contrast",  # alias often appearing as "ATC"
    "Miscellaneous area",
    "Rock outcrop",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns and harmonize common restriction field names."""
    df = df.copy()
    rename = {c: c.lower() for c in df.columns}
    df.rename(columns=rename, inplace=True)

    for target, aliases in _RESTRICTION_ALIASES.items():
        present = [c for c in df.columns if c in aliases]
        if present and present[0] != target:
            df.rename(columns={present[0]: target}, inplace=True)

    return df


def _standardize_reskind(text: str | float) -> str:
    """Normalize restriction kind strings (trim, title-case-ish, common alias fixes)."""
    if not isinstance(text, str):
        return ""
    x = text.strip()
    # light normalization
    x_low = x.lower()
    if "rock outcrop" in x_low:
        return "Rock outcrop"
    if "lithic" in x_low and "bedrock" in x_low:
        return "Lithic bedrock"
    if "paralithic" in x_low and "bedrock" in x_low:
        return "Paralithic bedrock"
    if "fragipan" in x_low:
        return "Fragipan"
    if "natric" in x_low:
        return "Natric"
    if "duripan" in x_low:
        return "Duripan"
    if "cement" in x_low:
        return "Cemented horizon"
    if "densic" in x_low and "contact" in x_low:
        return "Densic contact"
    if "strongly" in x_low and "textur" in x_low:
        return "Strongly contrasting textural stratification"
    if x_low in {"atc", "adverse texture contrast"}:
        return "Adverse texture contrast"
    if "misc" in x_low:
        return "Miscellaneous area"
    # default: return as-is with basic capitalization
    return x


# ---------------------------------------------------------------------
# 1) Load & align restrictions to components in study area
# ---------------------------------------------------------------------

def load_and_process_restrictions(
    restrictions_path: str,
    comp_data_mo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load restrictions and align to study-area components.

    Parameters
    ----------
    restrictions_path : str
        Path to 'corestrictions.csv' or equivalent.
    comp_data_mo : pd.DataFrame
        Component table filtered to your study area.
        Must contain ['cokey','mukey','comppct_r'] (and ideally 'majcompflag','compname').

    Returns
    -------
    pd.DataFrame
        Restrictions aligned to components with normalized columns:
        ['cokey','mukey','reskind','resdept_r','resdepb_r','comppct_r', ...]
        Rows restricted to those cokeys present in comp_data_mo.
    """
    df = pd.read_csv(restrictions_path, low_memory=False)
    df = _normalize_columns(df)

    required = {"cokey", "reskind"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Restrictions file missing required columns: {sorted(missing)}")

    # Optional numeric depth columns; coerce if present
    for c in ["resdept_r", "resdepb_r"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize strings
    df["reskind"] = df["reskind"].map(_standardize_reskind)

    # Limit to components in MO (or your study area)
    comp = comp_data_mo[["cokey", "mukey", "comppct_r"]].copy()
    comp["cokey"] = comp["cokey"].astype(str)
    comp["mukey"] = comp["mukey"].astype(str)

    df["cokey"] = df["cokey"].astype(str)
    out = df.merge(comp, on="cokey", how="inner")

    # Keep essentials + any pass-through columns you might need
    keep = ["cokey", "mukey", "reskind", "resdept_r", "resdepb_r", "comppct_r"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    log.info("Loaded restrictions: %s rows (aligned to study area)", len(out))
    return out


# ---------------------------------------------------------------------
# 2) Aggregate restriction depths by MU
# ---------------------------------------------------------------------

def aggregate_restriction_depths(restrictions_mo: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aggregate restriction depths per mukey for each restriction kind.

    For each reskind, returns a DataFrame with columns:
        ['mukey', 'resdept_min', 'resdepb_min', 'resdept_mean', 'resdepb_mean', 'count']

    Notes
    -----
    - We use both min and mean to give you choices downstream (you can choose min
      as conservative depth to first contact, or mean for typical depth).
    - Rows with entirely missing depths are ignored for stats; count is number of
      components with any depth info for that reskind in the MU.
    """
    if restrictions_mo.empty:
        return {}

    df = restrictions_mo.copy()
    df["mukey"] = df["mukey"].astype(str)

    out: Dict[str, pd.DataFrame] = {}
    for kind, grp in df.groupby("reskind", dropna=True):
        # ignore kinds that are empty strings after normalization
        if not isinstance(kind, str) or kind.strip() == "":
            continue

        # prepare grouped stats per MU
        g = grp.copy()
        # Only use rows with at least one depth value
        mask_any = g["resdept_r"].notna() | g["resdepb_r"].notna()
        g = g.loc[mask_any]
        if g.empty:
            continue

        agg = g.groupby("mukey").agg(
            resdept_min=("resdept_r", "min"),
            resdepb_min=("resdepb_r", "min"),
            resdept_mean=("resdept_r", "mean"),
            resdepb_mean=("resdepb_r", "mean"),
            count=("cokey", "nunique"),
        ).reset_index()

        out[kind] = agg.sort_values("mukey").reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# 3) Summaries of restriction kinds (per component, per MU)
# ---------------------------------------------------------------------

def create_restriction_summary(
    restrictions_mo: pd.DataFrame,
    comp_data_mo: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize restriction kinds per component (cokey) and per MU (mukey).

    Parameters
    ----------
    restrictions_mo : pd.DataFrame
        Output of `load_and_process_restrictions`.
    comp_data_mo : pd.DataFrame
        Component table with at least ['cokey','mukey','comppct_r'].

    Returns
    -------
    (reskinds_by_cokey, reskinds_by_mukey)
        reskinds_by_cokey : DataFrame with ['cokey','reskinds'] string list
        reskinds_by_mukey : DataFrame with ['mukey','reskinds'] string list
    """
    if restrictions_mo.empty:
        return (
            pd.DataFrame(columns=["cokey", "reskinds"]),
            pd.DataFrame(columns=["mukey", "reskinds"]),
        )

    df = restrictions_mo.copy()
    df["mukey"] = df["mukey"].astype(str)
    df["cokey"] = df["cokey"].astype(str)

    # Aggregate unique, non-empty kinds per cokey
    def _join_unique(series: pd.Series) -> str:
        vals = [str(x) for x in series.dropna().tolist() if str(x).strip() != ""]
        seen, out = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return ", ".join(out)

    by_cokey = (
        df.groupby("cokey")["reskind"]
        .apply(_join_unique)
        .reset_index(name="reskinds")
    )

    # Per MU: collect kinds across components in that MU
    by_mukey = (
        df.groupby("mukey")["reskind"]
        .apply(_join_unique)
        .reset_index(name="reskinds")
    )

    return by_cokey, by_mukey


# ---------------------------------------------------------------------
# 4) Component-percent coverage per restriction kind
# ---------------------------------------------------------------------

def calculate_restriction_percentages(
    comp_data_mo: pd.DataFrame,
    reskinds_by_cokey: pd.DataFrame,
    kinds_of_interest: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each restriction kind in `kinds_of_interest`, compute the total component
    percentage per mukey where that restriction kind is present.

    Parameters
    ----------
    comp_data_mo : pd.DataFrame
        Must include ['cokey','mukey','comppct_r'].
    reskinds_by_cokey : pd.DataFrame
        Must include ['cokey','reskinds'], where reskinds is a comma-separated list.
    kinds_of_interest : list[str] | None
        Restriction kinds to evaluate. If None, uses a default list.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Each key is a restriction kind; each value is a DataFrame:
            ['mukey','compct_sum']  (sum of component % for components in that MU
                                     whose reskinds contain the target kind)

    Notes
    -----
    - If your comppct_r is 0..1 instead of 0..100, compct_sum will also be 0..1.
    - Matching is case-insensitive substring match.
    """
    if kinds_of_interest is None:
        kinds_of_interest = _DEFAULT_KINDS_OF_INTEREST

    # Basic input checks
    req_comp = {"cokey", "mukey", "comppct_r"}
    if not req_comp <= set(comp_data_mo.columns):
        missing = req_comp - set(comp_data_mo.columns)
        raise KeyError(f"comp_data_mo missing required columns: {sorted(missing)}")

    req_res = {"cokey", "reskinds"}
    if not req_res <= set(reskinds_by_cokey.columns):
        missing = req_res - set(reskinds_by_cokey.columns)
        raise KeyError(f"reskinds_by_cokey missing required columns: {sorted(missing)}")

    comp = comp_data_mo[["cokey", "mukey", "comppct_r"]].copy()
    comp["cokey"] = comp["cokey"].astype(str)
    comp["mukey"] = comp["mukey"].astype(str)

    res = reskinds_by_cokey[["cokey", "reskinds"]].copy()
    res["cokey"] = res["cokey"].astype(str)

    # Join to get reskinds per component row
    joined = comp.merge(res, on="cokey", how="left")

    out: Dict[str, pd.DataFrame] = {}
    for kind in kinds_of_interest:
        if not isinstance(kind, str) or not kind.strip():
            continue
        pattern = kind.strip().lower()

        has_kind = joined["reskinds"].fillna("").str.lower().str.contains(pattern, na=False)
        subset = joined.loc[has_kind, ["mukey", "comppct_r"]]
        if subset.empty:
            out[kind] = pd.DataFrame(columns=["mukey", "compct_sum"])
            continue

        agg = (
            subset.groupby("mukey")["comppct_r"]
            .sum()
            .reset_index(name="compct_sum")
            .sort_values("compct_sum", ascending=False)
            .reset_index(drop=True)
        )
        out[kind] = agg

    return out
