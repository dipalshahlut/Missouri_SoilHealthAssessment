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
    

def wtd_mean(df, y):
    """
    Calculate the weighted mean using horizon thickness as weights.

    Args:
        df (pd.DataFrame): DataFrame containing the horizon data.
        y (str): Column name for the variable to compute the weighted mean.

    Returns:
        float: The weighted mean value, accounting for missing data.
    """
    # Calculate horizon thickness
    thickness = df['hzdepb_r'] - df['hzdept_r']

    # Extract the variable of interest
    values = df[y]

    # Compute the weighted mean, ignoring NaN values
    weighted_mean = np.average(values, weights=thickness, returned=False)

    return weighted_mean

def kgOrgC_sum(df, slice_it=False, depth=None, rm_nas=True, om_to_c=1.72):
    """
    #     Estimate total organic carbon content across horizons.

    #     Formula (as used in your codebase)
    #     ----------------------------------
    #     (thickness_cm / 10) * (OM / om_to_c) * bulk_density * (1 - fragment_vol%)
    #     where:
    #       thickness_cm = hzdepb_r - hzdept_r
    #       OM ~ organic matter
    #       bulk_density ~ dbthirdbar_r
    #       fragment_vol% ~ fragvol_r (0..100)

    #     Parameters
    #     ----------
    #     slice_it : bool
    #         If True, only consider the first `depth` horizons (row-wise).
    #     depth : int | None
    #         Number of horizons to include if `slice_it` is True.
    #     rm_nas : bool
    #         If True, ignore NaNs using np.nansum.
    #     om_to_c : float
    #         Conversion factor OM ➜ OC.

    #     Returns
    #     -------
    #     float
    #         Total organic C estimate (units consistent with inputs).
    """
    # Slice the DataFrame if needed
    if slice_it and depth is not None:
        df = df.iloc[:depth]

    # Calculate horizon thickness
    thickness = df['hzdepb_r'] - df['hzdept_r']

    # Calculate total organic carbon content
    organic_carbon = (thickness / 10) * (df['om_r'] / om_to_c) * df['dbthirdbar_r'] * (1 - df['fragvol_r'] / 100)
   
    # Sum the values, handling NaNs
    if rm_nas:
        total_organic_carbon = np.nansum(organic_carbon)
    else:
        total_organic_carbon = np.sum(organic_carbon)
   
    return total_organic_carbon

def awc_sum(df, rm_nas=True):
    """
    Calculate the total available water capacity (AWC) for soil horizons.

    Args:
        df (pd.DataFrame): DataFrame containing horizon-level data.
        rm_nas (bool): Whether to ignore NaN values in the calculation.

    Returns:
        float: Total available water capacity.
    """
    # Calculate horizon thickness
    thickness = df['hzdepb_r'] - df['hzdept_r']

    # Calculate AWC contribution for each horizon
    awc_contribution = thickness * df['awc_r']

    # Sum the AWC values, handling NaNs
    if rm_nas:
        total_awc = np.nansum(awc_contribution)
    else:
        total_awc = np.sum(awc_contribution)
   
    return total_awc

# # ---------------------------------------------------------------------
# # Aggregation helpers
# # ---------------------------------------------------------------------

def horizon_to_comp(horizon_df, depth, comp_df, 
                    vars_of_interest=None, 
                    varnames=None):
    """
    
    Args:
        horizon_df (DataFrame): DataFrame containing horizon-level data.
        depth (int): Depth in cm to filter the horizon data.
        comp_df (DataFrame): DataFrame containing component-level data.
        vars_of_interest (list): Horizon variables to aggregate.
        varnames (list): Corresponding variable names for the output.
    
    Returns:
        DataFrame: Aggregated data by component.
    """
    # Default values for vars_of_interest and varnames
    if vars_of_interest is None:
        vars_of_interest = [
            'claytotal_r', 'silttotal_r', 'sandtotal_r', 'om_r', 'cec7_r', 
            'dbthirdbar_r', 'fragvol_r', 'kwfact', 'ec_r', 'ph1to1h2o_r', 
            'sar_r', 'caco3_r', 'gypsum_r', 'lep_r', 'ksat_r'
        ]
    if varnames is None:
        varnames = [
            'clay', 'silt', 'sand', 'om', 'cec', 'bd', 'frags', 'kwf', 
            'ec', 'pH', 'sar', 'caco3', 'gyp', 'lep', 'ksat'
        ]

    # Generate column names
    columnames = [f"{var}_{depth}cm" for var in varnames]
    #print(pd.DataFrame({'vars_of_interest': vars_of_interest, 'columnames': columnames}))

    # Slice the DataFrame to include horizons up to the specified depth
    sliced_df = horizon_df[horizon_df['hzdepb_r'] <= depth].copy()
    # Initialize results
    result = {}

    # Weighted mean function
    def weighted_mean(data, var, weight):
        return np.average(data[var], weights=weight, axis=0)

    # Apply weighted mean for each variable of interest
    for var, colname in zip(vars_of_interest, columnames): #iterates through each variable in vars_of_interest
        if var in sliced_df.columns:
            # results stored in the site-level data 
            result[colname] = (
                sliced_df.groupby('cokey').apply(
                    lambda group: weighted_mean(group, var, group['hzdepb_r'] - group['hzdept_r'])
                )
            )
    # Additional aggregations (kgOrg and awc)
    result[f'kgOrg.m2_{depth}cm'] = (
        sliced_df.groupby('cokey').apply(lambda group: kgOrgC_sum(group))
    )
    result[f'awc_{depth}cm'] = (
        sliced_df.groupby('cokey').apply(lambda group: awc_sum(group))
    )

    # Convert result to a DataFrame
    result_df = pd.DataFrame(result).reset_index()
    
    # Merge component-level matadata to site-level data
    result_df = result_df.merge(comp_df[['cokey', 'compname', 'mukey', 'comppct_r']], on='cokey', how='left')
   
    # Rearrange columns for final output
    result_df = result_df[['mukey', 'cokey', 'compname', 'comppct_r'] + columnames + 
                           [f'kgOrg.m2_{depth}cm', f'awc_{depth}cm']]
    
    # Rename specific columns
    result_df.rename(columns={'comppct_r': 'comppct'}, inplace=True)

    return result_df


def MUaggregate(df, varname):
    result = {}
    grouped = df.groupby('mukey')
    for name, group in grouped:
        valid_data = group[varname].notna()
        if valid_data.sum() == 0:
            result[name] = np.nan
        else:
            weights = group.loc[valid_data, 'comppct']
            values = group.loc[valid_data, varname]
            weighted_avg = np.average(values, weights=weights)
            result[name] = weighted_avg
    return result

def MUAggregate_wrapper(df, varnames):
    results = {varname: MUaggregate(df, varname) for varname in varnames}
    result_df = pd.DataFrame(results)
    result_df['mukey'] = result_df.index.astype(int)
    return result_df.reset_index(drop=True)

# # ---------------------------------------------------------------------
# # Misc helpers
# # ---------------------------------------------------------------------
def concat_names(x, decat=False):
    """
    Concatenates unique names from a list, handling NaN values.
    If `decat` is True, splits input strings by '-' before processing.
    """
    if decat:
        # Split each string by '-' and flatten the list
        x = [item for sublist in [str(i).split('-') for i in x if pd.notnull(i)] for item in sublist]

    # Handle the case where all elements are NaN
    if np.all(pd.isnull(x)):
        return np.nan
 
    # Filter out NaN values and get unique values
    unique_values = sorted(set([i for i in x if pd.notnull(i)]))

    # If there is only one unique value, return it
    if len(unique_values) == 1:
        return unique_values[0]

    # Otherwise, concatenate unique values with '-'
    return '-'.join(unique_values)

def reskind_comppct(reskind, comp_d, reskind_df):
    filtered = comp_d[comp_d['cokey'].isin(reskind_df.loc[reskind_df['reskinds'].str.contains(reskind, case=False, na=False), 'cokey'])]
    result = filtered.groupby('mukey')['comppct_r'].sum().reset_index(name='compct_sum')
    return result


