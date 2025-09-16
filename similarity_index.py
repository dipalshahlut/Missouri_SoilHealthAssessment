#!/usr/bin/env python3
"""
similarity_index.py — Compare two clustering label CSVs and report similarity.

Behavior
- Reads two CSVs containing cluster labels (from --input-dir).
- Pick which two CSVs to compare in this order:
    1) Explicit --file-a / --file-b
    2) Derived from (--method-a, --k-a) / (--method-b, --k-b)
    3) Auto-pick the two most recent matching:
         *clusters_vae_algorithms_merged_*_k*.csv
- Auto-detect label columns unless you pass --col-a / --col-b,
  or it derives "Method_bestK" from (--method-*, --k-*).
- Writes outputs (counts CSV, row% CSV, HTML, PNG, META) to --output-dir
  (default: same as --input-dir).

Outputs (written to output-dir):
  - <stemA>__vs__<stemB>_crosstab_counts.csv
  - <stemA>__vs__<stemB>_crosstab_rowpct.csv
  - <stemA>__vs__<stemB>_table.html
  - <stemA>__vs__<stemB>_table.png
  - <stemA>__vs__<stemB>_meta.json

Usage:
python similarity_index.py \
  --input-dir /Users/me/SHA_copy/data/aggResult/shapefiles_with_data \
  --file-a MO_30cm_clusters_vae_algorithms_merged_KMeans_k10.csv \
  --file-b MO_30cm_clusters_vae_algorithms_merged_KMeans_k12.csv
or 
python similarity_index.py \
  --input-dir /Users/me/SHA_copy/data/aggResult/shapefiles_with_data \
  --output-dir /Users/me/SHA_copy/data/aggResult/compare_out \
  --method-a KMeans --k-a 10 \
  --method-b GMM --k-b 12
"""

from pathlib import Path
import argparse
import re, sys, os
import json
from typing import Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt      # backend for dataframe_image
import dataframe_image as dfi        # export styled DataFrame to PNG

# Default pattern subfolder often used by your pipeline; optional
SUBDIR_DEFAULT = "shapefiles_with_data"

# --------------------------- logging ---------------------------

def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.INFO
    if verbosity >= 3:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# Canonicalize method names → "KMeans", "Agglomerative", "Birch", "GMM"
_CANON = {"kmeans": "KMeans", "agglomerative": "Agglomerative",
          "birch": "Birch", "gmm": "GMM"}

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def label_col(method: str, k: int) -> str:
    """Build a canonical label column name like 'KMeans_best10'."""
    m = _CANON.get(str(method).strip().lower(), str(method).strip())
    return f"{m}_best{int(k)}"

def _find_label_column(df: pd.DataFrame) -> str:
    """Heuristic to find a label column."""
    # explicit names first
    for name in df.columns:
        if str(name).strip().lower() in {"cluster", "labels", "label"}:
            return name
    # pattern Method_bestK
    pat = re.compile(r".*_best\d+$", flags=re.IGNORECASE)
    for name in df.columns:
        if pat.match(str(name)):
            return name
    # small-cardinality numeric-ish
    for name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
        uniq = pd.unique(s.dropna().astype(int))
        if 2 <= len(uniq) <= 100:
            return name
    raise ValueError("Could not auto-detect a label column. Pass --col-a/--col-b or method/k.")

def _normalize_key_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Prefer MUKEY/objectid/geoid; else create positional key."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ["mukey", "objectid", "geoid"]:
        if candidate in lower_map:
            key = lower_map[candidate]
            out = df.copy()
            out[key] = out[key].astype(str)
            return out, key
    out = df.copy()
    out["row_idx"] = np.arange(len(out))
    return out, "row_idx"

def _load_labels(csv_path: Path, col_hint: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Return (compact_df['key','labels'], key_name)."""
    df_raw = pd.read_csv(csv_path)
    col = col_hint or _find_label_column(df_raw)
    s = pd.to_numeric(df_raw[col], errors="coerce").dropna().astype(int)

    keyed, key_name = _normalize_key_columns(df_raw)
    key_series = keyed[key_name].astype(str)

    compact = pd.DataFrame({"key": key_series, "labels": s}).dropna()
    compact["key"] = compact["key"].astype(str)
    return compact, key_name

def _glob_latest_two(io_dir: Path) -> Tuple[Path, Path]:
    """Pick two most recent CSVs matching the spatial_maps name pattern."""
    cands = sorted(
        io_dir.glob("*clusters_vae_algorithms_merged_*_k*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(cands) < 2:
        raise FileNotFoundError(f"Need at least two CSVs in {io_dir} matching '*clusters_vae_algorithms_merged_*_k*.csv'")
    return cands[0], cands[1]

def _resolve_from_method_k(io_dir: Path, method: Optional[str], k: Optional[int]) -> Optional[Path]:
    """Resolve a CSV by method/k if given; else None. Robust to arbitrary prefixes."""
    if method is None or k is None:
        return None
    m = _CANON.get(str(method).strip().lower(), str(method).strip())
    tail = f"_clusters_vae_algorithms_merged_{m}_k{int(k)}.csv"
    matches = list(io_dir.glob(f"*{tail}"))
    if not matches:
        raise FileNotFoundError(f"No CSV found in {io_dir} matching '*{tail}'")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def _choose_common_key(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[str, str, bool]:
    """
    Return (key_a, key_b, requires_agg).
    Prefer OBJECTID if present on both and unique; else fall back to MUKEY/mukey.
    If we fall back to mukey and it's not unique, requires_agg = True (group to mode).
    """
    # Find case-insensitive column name matches
    def find(df, name):
        for c in df.columns:
            if str(c).lower() == name.lower():
                return c
        return None

    # 1) Try OBJECTID
    a_obj = find(df_a, "OBJECTID")
    b_obj = find(df_b, "OBJECTID")
    if a_obj and b_obj:
        if df_a[a_obj].nunique() == len(df_a) and df_b[b_obj].nunique() == len(df_b):
            return a_obj, b_obj, False  # unique on both

    # 2) Try MUKEY/mukey
    a_mu = find(df_a, "mukey") or find(df_a, "MUKEY")
    b_mu = find(df_b, "mukey") or find(df_b, "MUKEY")
    if a_mu and b_mu:
        # If both are unique, no aggregation needed; otherwise aggregate to mode
        need_agg = not (df_a[a_mu].nunique() == len(df_a) and df_b[b_mu].nunique() == len(df_b))
        return a_mu, b_mu, need_agg

    raise KeyError("Could not find a common join key (OBJECTID or MUKEY/mukey) in both CSVs.")


def _mode_int(series: pd.Series) -> float | int | None:
    """Return the most frequent integer value in a series (NaN if none)."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if s.empty:
        return np.nan
    # For ties, value_counts() picks the smallest label by default order
    return int(s.value_counts().idxmax())


def _compact_labels(df_raw: pd.DataFrame, key: str, label_col: str, agg_to_mode: bool) -> pd.DataFrame:
    """
    Return a compact table with columns ['key', 'labels'].
    If agg_to_mode=True, group by key and take modal label.
    """
    if label_col not in df_raw.columns:
        label_col = _find_label_column(df_raw)

    if agg_to_mode:
        out = (
            df_raw.groupby(key, dropna=False)[label_col]
                  .apply(_mode_int)
                  .dropna()
                  .astype(int)
                  .reset_index()
                  .rename(columns={key: "key", label_col: "labels"})
        )
    else:
        out = df_raw[[key, label_col]].copy()
        out["labels"] = pd.to_numeric(out[label_col], errors="coerce")
        out = out.dropna(subset=["labels"])
        out["labels"] = out["labels"].astype(int)
        out = out.rename(columns={key: "key"})[["key", "labels"]]

    out["key"] = out["key"].astype(str)
    return out


# ---- Core API --------------------------------------------------------------

def run_similarity_index(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    method_a: Optional[str] = None,
    k_a: Optional[int] = None,
    method_b: Optional[str] = None,
    k_b: Optional[int] = None,
    col_a: Optional[str] = None,
    col_b: Optional[str] = None,
) -> dict:
    """
    Compute similarity (ARI), counts table, and row% table between two label CSVs.
    Robust to non-unique keys (aggregates to modal label per key when joining by MUKEY).
    """
    io_dir = Path(input_dir)
    if not io_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {io_dir}")
    out_dir = _ensure_dir(Path(output_dir) if output_dir else io_dir)

    # Resolve CSV paths
    path_a: Optional[Path] = (io_dir / file_a) if file_a else _resolve_from_method_k(io_dir, method_a, k_a)
    path_b: Optional[Path] = (io_dir / file_b) if file_b else _resolve_from_method_k(io_dir, method_b, k_b)

    if path_a is None or path_b is None:
        latest_a, latest_b = _glob_latest_two(io_dir)
        path_a = path_a or latest_a
        path_b = path_b or (latest_b if latest_b != path_a else latest_a)

    if not path_a.exists():
        raise FileNotFoundError(f"CSV A not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"CSV B not found: {path_b}")

    # Read with low_memory=False to avoid DtypeWarning
    df_raw_a = pd.read_csv(path_a, low_memory=False)
    df_raw_b = pd.read_csv(path_b, low_memory=False)

    # If user provided method/k but no column, derive column names
    if col_a is None and method_a is not None and k_a is not None:
        col_a = label_col(method_a, k_a)
    if col_b is None and method_b is not None and k_b is not None:
        col_b = label_col(method_b, k_b)

    # Choose a common key; aggregate to mode per key if necessary
    key_a, key_b, agg_needed = _choose_common_key(df_raw_a, df_raw_b)

    comp_a = _compact_labels(df_raw_a, key=key_a, label_col=(col_a or _find_label_column(df_raw_a)), agg_to_mode=agg_needed)
    comp_b = _compact_labels(df_raw_b, key=key_b, label_col=(col_b or _find_label_column(df_raw_b)), agg_to_mode=agg_needed)

    # Ensure 1:1 by dropping any accidental duplicates after compaction
    comp_a = comp_a.drop_duplicates(subset=["key"], keep="first")
    comp_b = comp_b.drop_duplicates(subset=["key"], keep="first")

    # 1:1 merge
    merged = (
        pd.merge(
            comp_a.rename(columns={"labels": "labels_a"}),
            comp_b.rename(columns={"labels": "labels_b"}),
            on="key", how="inner", validate="1:1"
        )
        .dropna(subset=["labels_a", "labels_b"])
        .astype({"labels_a": int, "labels_b": int})
    )

    # Metric
    ari = float(adjusted_rand_score(merged["labels_a"], merged["labels_b"]))

    # Crosstabs (display 1-based cluster ids)
    disp_a = merged["labels_a"] + 1
    disp_b = merged["labels_b"] + 1
    counts = pd.crosstab(disp_a, disp_b, dropna=False)
    rowpct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(3)

    # Output paths
    stem_a = path_a.stem
    stem_b = path_b.stem
    prefix = f"{stem_a}__vs__{stem_b}"

    counts_csv = out_dir / f"{prefix}_crosstab_counts.csv"
    rowpct_csv = out_dir / f"{prefix}_crosstab_rowpct.csv"
    html_path  = out_dir / f"{prefix}_table.html"
    png_path   = out_dir / f"{prefix}_table.png"
    meta_path  = out_dir / f"{prefix}_meta.json"

    counts.to_csv(counts_csv, index=True)
    rowpct.to_csv(rowpct_csv, index=True)

    styled = (
        counts.style
              .format("{:,}")
              .background_gradient(cmap="Blues", axis=None)
              .set_caption(f"Cross-tab: {stem_a} vs {stem_b}")
              .set_table_styles([
                  {"selector": "caption", "props": "caption-side:top; font-weight:bold; font-size:12pt;"},
                  {"selector": "th", "props": "font-weight:bold; text-align:center; padding:6px;"},
                  {"selector": "td", "props": "text-align:center; padding:6px;"},
              ])
    )
    styled.to_html(html_path)
    try:
        dfi.export(styled, png_path, table_conversion="matplotlib", dpi=300)
    except Exception as e:
        print(f"⚠️  PNG export failed: {e}")

    meta = {
        "file_a": str(path_a), "file_b": str(path_b),
        "key_a": key_a, "key_b": key_b,
        "aggregated_by_key": bool(agg_needed),
        "n_aligned": int(len(merged)),
        "ari": ari,
        "outputs_dir": str(out_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"📂 input : {io_dir}")
    print(f"💾 output: {out_dir}")
    print(f"🔑 join  : {key_a} ↔ {key_b}  (agg={agg_needed})")
    print(f"🧮 ARI = {ari:.4f}")
    print(f"↳ counts : {counts_csv.name}")
    print(f"↳ rowpct : {rowpct_csv.name}")
    print(f"↳ html   : {html_path.name}")
    print(f"↳ png    : {png_path.name}")

    return {
        "ari": ari,
        "counts_csv": counts_csv,
        "rowpct_csv": rowpct_csv,
        "html": html_path,
        "png": png_path,
        "meta": meta_path,
        "out_dir": out_dir,
    }

# --- CLI --------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two clustering label CSVs and report similarity (ARI + crosstab)."
    )

    # Directories
    p.add_argument(
        "--input-dir", type=Path, required=False,
        help="Directory containing the label CSVs. If omitted, uses --output-dir/--subdir."
    )
    p.add_argument(
        "--output-dir", type=Path, required=False,
        help="Directory to write outputs. Defaults to --input-dir if not provided."
    )
    p.add_argument(
        "--subdir", type=str, default=SUBDIR_DEFAULT,
        help=f"If --input-dir not given, read from <output-dir>/<subdir>. Default: {SUBDIR_DEFAULT}"
    )

    # Option A: pass filenames directly
    p.add_argument("--file-a", type=str, help="Filename of first CSV (under input-dir).")
    p.add_argument("--file-b", type=str, help="Filename of second CSV (under input-dir).")

    # Option B: derive filenames from method/k
    p.add_argument("--method-a", type=str, help="Method for A (e.g., KMeans, GMM).")
    p.add_argument("--k-a", type=int, help="k for A (e.g., 10).")
    p.add_argument("--method-b", type=str, help="Method for B (e.g., KMeans, GMM).")
    p.add_argument("--k-b", type=int, help="k for B (e.g., 12).")

    # Optional: force label column names
    p.add_argument("--col-a", type=str, help="Explicit label column in CSV A (e.g., KMeans_best10).")
    p.add_argument("--col-b", type=str, help="Explicit label column in CSV B (e.g., KMeans_best12).")

    # Verbosity (optional)
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")

    return p


def main() -> None:
    args = _build_argparser().parse_args()
    _setup_logging(args.verbose)

    # Resolve input/output dirs:
    # - If --input-dir provided, use it. Else use <output-dir>/<subdir>.
    if args.input_dir:
        input_dir = args.input_dir
        output_dir = args.output_dir or args.input_dir
    else:
        if not args.output_dir:
            raise SystemExit("You must provide either --input-dir OR --output-dir (with optional --subdir).")
        input_dir = args.output_dir / args.subdir
        output_dir = args.output_dir

    run_similarity_index(
        input_dir=input_dir,
        output_dir=output_dir,
        file_a=args.file_a,
        file_b=args.file_b,
        method_a=args.method_a,
        k_a=args.k_a,
        method_b=args.method_b,
        k_b=args.k_b,
        col_a=args.col_a,
        col_b=args.col_b,
    )
    print("✅ Similarity index completed successfully.")
    print("Results saved in:", args.output_dir)

if __name__ == "__main__":
    sys.exit(main())
