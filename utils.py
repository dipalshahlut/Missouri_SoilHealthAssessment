#!/usr/bin/env python3
# utils.py
"""
Shared utilities for the SSURGO/soil-processing pipeline.

Contents
--------
Logging:
    - setup_logging(): root logger with console + rotating file handlers
    - time_this: decorator to log start/finish (with duration) around any function
    - get_task_logger(): module-named logger used by time_this

Horizon-level utilities:
    - wtd_mean(): weighted mean over horizon slices (weights = thickness)
    - kgOrgC_sum(): estimate organic C stock across horizons
    - awc_sum(): total available water capacity across horizons

Aggregation helpers:
    - horizon_to_comp(): horizon ➜ component aggregation for a given depth slice
    - MUaggregate(): component ➜ map unit weighted aggregation
    - MUAggregate_wrapper(): apply MUaggregate over multiple variables

Misc helpers:
    - concat_names(): join unique items into a readable, comma-separated string
    - reskind_comppct(): % of component composition per MU for a restriction kind
"""


import functools
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logging(
    log_file: str = "workflow.log",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure the root logger with both console and rotating file handlers.

    Parameters
    ----------
    log_file : str
        Log filename (created in `log_dir`).
    log_dir : str | None
        Directory for the log file. If None, use the calling script's directory.
    level : int
        Root logger threshold (e.g., logging.INFO).
    max_bytes : int
        Max size per log file before rotation.
    backup_count : int
        Number of rotated backups to keep.
    console_level : int
        Level for console output.
    file_level : int
        Level for file output.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    # Determine a sensible log directory
    if log_dir is None:
        try:
            frame = sys._getframe(1)  # caller
            script_path = frame.f_globals.get("__file__", ".")
            log_dir = os.path.dirname(os.path.abspath(script_path))
        except Exception:
            log_dir = os.getcwd()

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        fmt = "%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Rotating file handler
        try:
            fh = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error("Failed to attach file handler: %s", e, exc_info=True)

        logger.info(
            "Logging configured. Root=%s | Console=%s | File=%s | FilePath=%s",
            logging.getLevelName(level),
            logging.getLevelName(console_level),
            logging.getLevelName(file_level),
            log_path,
        )

    return logger


def get_task_logger(func: Callable) -> logging.Logger:
    """
    Return a logger named after the module owning `func`.
    Helpful to group logs by file (e.g., 'horizon_processing').

    Example
    -------
    logger = get_task_logger(my_function)
    logger.info("message")
    """
    module_path = func.__globals__.get("__file__", "unknown_module")
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    return logging.getLogger(module_name)


def time_this(func: Callable) -> Callable:
    """
    Decorator: log start/finish (with duration) for `func`.
    Logs exceptions with traceback and re-raises.

    Usage
    -----
    @time_this
    def step(...):
        ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_task_logger(func)
        name = func.__name__
        logger.info("Starting: %s ...", name)
        status = "completed successfully"
        import time as _time
        t0 = _time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            status = f"failed with {type(e).__name__}"
            logger.error("Exception in %s: %s", name, e, exc_info=True)
            raise
        finally:
            dt = _time.perf_counter() - t0
            lvl = logging.INFO if status == "completed successfully" else logging.ERROR
            logger.log(lvl, "Finished: %s %s in %.4f seconds.", name, status, dt)
    return wrapper


# ---------------------------------------------------------------------
# Horizon-level utilities
# ---------------------------------------------------------------------

def _horizon_thickness(df: pd.DataFrame) -> np.ndarray:
    """Return horizon thickness (hzdepb_r - hzdept_r)."""
    return (df["hzdepb_r"] - df["hzdept_r"]).to_numpy()


def wtd_mean(df: pd.DataFrame, y: str) -> float:
    """
    Weighted mean over horizons using thickness as weights.

    Parameters
    ----------
    df : pd.DataFrame
        Horizon dataframe with columns: hzdept_r, hzdepb_r, and `y`.
    y : str
        Column to average.

    Returns
    -------
    float
        Weighted mean. NaNs in `y` are ignored by masking weights/values.
    """
    values = df[y].to_numpy()
    weights = _horizon_thickness(df)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if mask.sum() == 0:
        return np.nan
    try:
        return float(np.average(values[mask], weights=weights[mask]))
    except ZeroDivisionError:
        return np.nan


def kgOrgC_sum(
    df: pd.DataFrame,
    slice_it: bool = False,
    depth: Optional[int] = None,
    rm_nas: bool = True,
    om_to_c: float = 1.72,
) -> float:
    """
    Estimate total organic carbon content across horizons.

    Formula (as used in your codebase)
    ----------------------------------
    (thickness_cm / 10) * (OM / om_to_c) * bulk_density * (1 - fragment_vol%)
    where:
      thickness_cm = hzdepb_r - hzdept_r
      OM ~ organic matter
      bulk_density ~ dbthirdbar_r
      fragment_vol% ~ fragvol_r (0..100)

    Parameters
    ----------
    slice_it : bool
        If True, only consider the first `depth` horizons (row-wise).
    depth : int | None
        Number of horizons to include if `slice_it` is True.
    rm_nas : bool
        If True, ignore NaNs using np.nansum.
    om_to_c : float
        Conversion factor OM ➜ OC.

    Returns
    -------
    float
        Total organic C estimate (units consistent with inputs).
    """
    d = df.iloc[:depth] if slice_it and depth is not None else df
    thickness_cm = (d["hzdepb_r"] - d["hzdept_r"]).astype(float)
    term = (
        (thickness_cm / 10.0)
        * (d["om_r"].astype(float) / om_to_c)
        * d["dbthirdbar_r"].astype(float)
        * (1.0 - d["fragvol_r"].astype(float) / 100.0)
    )
    return float(np.nansum(term) if rm_nas else np.sum(term))


def awc_sum(df: pd.DataFrame, rm_nas: bool = True) -> float:
    """
    Total Available Water Capacity (AWC) across horizons.

    Requires: hzdept_r, hzdepb_r, awc_r (if missing, returns NaN gracefully).
    """
    if "awc_r" not in df.columns:
        return np.nan  # no AWC in this dataset

    thickness_cm = (df["hzdepb_r"] - df["hzdept_r"]).astype(float)
    awc = pd.to_numeric(df["awc_r"], errors="coerce")
    term = thickness_cm * awc
    return float(np.nansum(term) if rm_nas else np.sum(term))


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------

def horizon_to_comp(
    horizon_df: pd.DataFrame,
    depth: int,
    comp_df: pd.DataFrame,
    vars_of_interest: Optional[List[str]] = None,
    varnames: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate horizon-level values up to component-level for a depth slice.

    Parameters
    ----------
    horizon_df : pd.DataFrame
        Horizon rows (must include: cokey, hzdept_r, hzdepb_r, variables).
    depth : int
        Include horizons whose bottom depth (hzdepb_r) <= depth (cm).
    comp_df : pd.DataFrame
        Component metadata (at least cokey, compname, mukey, comppct_r).
    vars_of_interest : list[str] | None
        Horizon variables to compute weighted averages for.
    varnames : list[str] | None
        Friendly output names aligned to `vars_of_interest`.

    Returns
    -------
    pd.DataFrame
        Component-level aggregates with:
          ['mukey','cokey','compname','comppct', <metrics...>, 'kgOrg.m2_{depth}cm','awc_{depth}cm']
    """
    if vars_of_interest is None:
        vars_of_interest = [
            "claytotal_r", "silttotal_r", "sandtotal_r", "om_r", "cec7_r",
            "dbthirdbar_r", "fragvol_r", "kwfact", "ec_r", "ph1to1h2o_r",
            "sar_r", "caco3_r", "gypsum_r", "lep_r", "ksat_r",
        ]
    if varnames is None:
        varnames = [
            "clay", "silt", "sand", "om", "cec", "bd", "frags", "kwf",
            "ec", "pH", "sar", "caco3", "gyp", "lep", "ksat",
        ]

    # Select horizons up to the requested depth threshold
    sliced = horizon_df[horizon_df["hzdepb_r"] <= depth].copy()

    # Weighted mean helper with NaN-safe masking
    def _wmean(group: pd.DataFrame, col: str) -> float:
        vals = group[col].to_numpy(dtype=float)
        wts = (group["hzdepb_r"] - group["hzdept_r"]).to_numpy(dtype=float)
        mask = ~np.isnan(vals) & ~np.isnan(wts)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(vals[mask], weights=wts[mask]))

    result: Dict[str, pd.Series] = {}
    out_cols = [f"{name}_{depth}cm" for name in varnames]

    for hz_col, out_col in zip(vars_of_interest, out_cols):
        if hz_col in sliced.columns:
            result[out_col] = sliced.groupby("cokey").apply(lambda g: _wmean(g, hz_col))

    # Additional aggregates
    result[f"kgOrg.m2_{depth}cm"] = sliced.groupby("cokey").apply(lambda g: kgOrgC_sum(g))
    result[f"awc_{depth}cm"] = sliced.groupby("cokey").apply(lambda g: awc_sum(g))

    comp_level = pd.DataFrame(result).reset_index()

    # Attach component metadata and tidy
    keep = ["cokey", "compname", "mukey", "comppct_r"]
    comp_meta = comp_df[keep].copy()
    comp_meta.rename(columns={"comppct_r": "comppct"}, inplace=True)

    out = comp_level.merge(comp_meta, on="cokey", how="left")
    # Reorder for readability
    out = out[["mukey", "cokey", "compname", "comppct"] + out_cols + [f"kgOrg.m2_{depth}cm", f"awc_{depth}cm"]]
    return out


def MUaggregate(df: pd.DataFrame, varname: str) -> Dict[str, float]:
    """
    Aggregate component-level values to the MU level using component % weights.

    Parameters
    ----------
    df : pd.DataFrame
        Component-level rows with columns: ['mukey', 'comppct', varname]
    varname : str
        Column to aggregate.

    Returns
    -------
    dict
        {mukey -> weighted average} (NaN if no valid components)
    """
    out: Dict[str, float] = {}
    for mukey, grp in df.groupby("mukey"):
        valid = grp[varname].notna()
        if valid.sum() == 0:
            out[mukey] = np.nan
        else:
            w = grp.loc[valid, "comppct"].astype(float)
            v = grp.loc[valid, varname].astype(float)
            try:
                out[mukey] = float(np.average(v, weights=w))
            except ZeroDivisionError:
                out[mukey] = np.nan
    return out


def MUAggregate_wrapper(df: pd.DataFrame, varnames: Iterable[str]) -> pd.DataFrame:
    """
    Apply MUaggregate across multiple variables and return an MU-wide dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Component-level table with at least ['mukey', 'comppct', *varnames]
    varnames : Iterable[str]
        Variables to aggregate to MU level.

    Returns
    -------
    pd.DataFrame
        One row per mukey with aggregated columns; includes 'mukey' as a column.
    """
    results = {var: MUaggregate(df, var) for var in varnames}
    mu_df = pd.DataFrame(results)
    mu_df["mukey"] = mu_df.index.astype(str)
    mu_df.reset_index(drop=True, inplace=True)
    return mu_df


# ---------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------

def concat_names(series: pd.Series, sep: str = ", ") -> str:
    """
    Join unique, non-null values in a readable way (stable order).

    Example
    -------
    concat_names(pd.Series(["A", "B", "A"])) -> "A, B"
    """
    vals = [str(x) for x in series.dropna().tolist() if str(x).strip() != ""]
    # Preserve original order but drop duplicates
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sep.join(out)


def reskind_comppct(reskind: str, comp_df: pd.DataFrame, reskinds_by_cokey: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total component % per MU for a given restriction kind.

    Parameters
    ----------
    reskind : str
        Restriction kind name to match (case-insensitive, substring match ok).
    comp_df : pd.DataFrame
        Component data with ['cokey','mukey','comppct_r'].
    reskinds_by_cokey : pd.DataFrame
        Table with ['cokey','reskinds'] listing kinds per component.

    Returns
    -------
    pd.DataFrame
        Columns: ['mukey','compct_sum'] where compct_sum is the sum of comppct_r
        over components whose reskinds contain `reskind`.
    """
    # Identify which components (cokey) have this restriction kind
    mask = reskinds_by_cokey["reskinds"].str.contains(reskind, case=False, na=False)
    target_cokeys = set(reskinds_by_cokey.loc[mask, "cokey"].tolist())
    filtered = comp_df[comp_df["cokey"].isin(target_cokeys)]
    out = (
        filtered.groupby("mukey")["comppct_r"]
        .sum()
        .reset_index(name="compct_sum")
    )
    return out
