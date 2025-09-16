# metric_plots.py
#!/usr/bin/env python3
"""
metrics_plots.py
Compute & save method-vs-k metrics on FEATURE SPACE (default: UN-SCALED):
  - Gap Statistic
  - Calinski–Harabasz
  - Silhouette
  - WSS (Inertia-like; computed post-hoc for all methods)

Outputs:
  - <output_dir>/figures/wss_curves.png
  - <output_dir>/figures/ch_curves.png
  - <output_dir>/figures/silhouette_curves.png
  - <output_dir>/figures/gap_curves.png
  - <output_dir>/metrics_<Method>.csv   (per-method table with k, wss, ch, sil, gap)

CLI:
  python metrics_plots.py -o /path/to/aggResult --k-min 2 --k-max 20 --gap-B 10
  # or force a particular file:
  python metrics_plots.py -o /path/to/aggResult --data-path /path/to/data_scaled.npy
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Callable

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score


# -----------------------
# Logging
# -----------------------
def _setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity >= 2: level = logging.INFO
    if verbosity >= 3: level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------
# Robust CSV reader (handles weird encodings)
# -----------------------
def _read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


# -----------------------
# IO (load UN-SCALED by default; fallback to scaled)
# -----------------------
def _load_feature_matrix(output_dir: str, data_path: Optional[str] = None) -> np.ndarray:
    """
    Load the feature matrix for clustering metrics.

    Priority when data_path is not given:
      1) <output_dir>/data_unscaled.npy
      2) <output_dir>/data_unscaled.csv
      3) <output_dir>/data_scaled.npy
      4) <output_dir>/data_scaled.csv
      5) <output_dir>/prepared_df.parquet  (numeric columns only)
    """
    # If user supplied an explicit path, use it (.npy, .csv, .parquet)
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".npy":
            X = np.load(data_path)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(data_path)
            X = df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        else:
            df = _read_csv_robust(data_path)
            X = df.select_dtypes(include=["number"]).to_numpy(dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Feature matrix must be 2D, got {X.shape}")
        return X

    # Autodetect in preferred order (UNSCALED first)
    candidates = [
        os.path.join(output_dir, "data_unscaled.npy"),
        os.path.join(output_dir, "data_unscaled.csv"),
        os.path.join(output_dir, "data_scaled.npy"),
        os.path.join(output_dir, "data_scaled.csv"),
        os.path.join(output_dir, "prepared_df.parquet"),
    ]
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    if not found:
        raise FileNotFoundError(
            "Could not find feature matrix. Looked for: "
            + ", ".join(candidates)
        )

    ext = os.path.splitext(found)[1].lower()
    if ext == ".npy":
        X = np.load(found)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(found)
        X = df.select_dtypes(include=["number"]).to_numpy(dtype=float)
    else:
        df = _read_csv_robust(found)
        X = df.select_dtypes(include=["number"]).to_numpy(dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got {X.shape}")
    logging.info("Loaded feature matrix from: %s (shape=%s)", found, X.shape)
    return X


# -----------------------
# Metrics helpers
# -----------------------
def _inertia_from_labels(X: np.ndarray, labels: np.ndarray) -> float:
    """Within-SS (WSS) computed from labels (works for any method)."""
    s = 0.0
    for u in np.unique(labels):
        pts = X[labels == u]
        if pts.size == 0:
            continue
        c = pts.mean(axis=0)
        s += ((pts - c) ** 2).sum()
    return float(s)


def _gap_statistic(
    X: np.ndarray,
    cluster_fn: Callable[[np.ndarray, int], np.ndarray],
    k: int,
    refs: List[np.ndarray]
) -> float:
    """Tibshirani et al. gap statistic: E[log(W*_k)] - log(W_k)."""
    labels = cluster_fn(X, k)
    Wk = _inertia_from_labels(X, labels)
    Wk_refs = []
    for Xr in refs:
        try:
            lr = cluster_fn(Xr, k)
            Wk_refs.append(_inertia_from_labels(Xr, lr))
        except Exception:
            Wk_refs.append(np.nan)
    return np.log(np.nanmean(Wk_refs)) - np.log(Wk)


def _methods_map(random_state: int = 42) -> Dict[str, Callable[[np.ndarray, int], np.ndarray]]:
    return {
        "KMeans":        lambda X, k: KMeans(n_clusters=k, random_state=random_state).fit_predict(X),
        "Agglomerative": lambda X, k: AgglomerativeClustering(n_clusters=k).fit_predict(X),
        "Birch":         lambda X, k: Birch(n_clusters=k).fit_predict(X),
        "GMM":           lambda X, k: GaussianMixture(n_components=k, random_state=random_state).fit_predict(X),
    }


# -----------------------
# Plot helper (one metric per figure)
# -----------------------
def _plot_metric_curves(
    ks: List[int],
    results: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    ylabel: str,
    out_path: str,
    logy: bool = False,
) -> str:
    plt.figure(figsize=(9, 4.8))
    for name in results:
        plt.plot(ks, results[name][metric_key], marker='o', label=name)
    if logy:
        plt.yscale("log")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(ncol=4, fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


# -----------------------
# Public API
# -----------------------
def run(
    base_dir: Optional[str] = None,           # kept for signature parity; unused
    output_dir: str = ".",
    data_path: Optional[str] = None,          # explicit path to unscaled/scaled .npy/.csv/.parquet
    methods: Optional[List[str]] = None,      # if None, use all four
    k_min: int = 2,
    k_max: int = 20,
    gap_B: int = 10,
    verbosity: int = 1,
    random_state: int = 42,
    # --------- backward-compat knobs ---------
    scaled_path: Optional[str] = None,        # accept the legacy kwarg from runbook
    **kwargs,
) -> Dict[str, str]:
    """
    Compute and plot WSS, Calinski–Harabasz, Silhouette, and Gap Statistic across k
    for multiple clustering methods using the **feature matrix** (UNSCALED by default).

    Saves (in <output_dir>/figures):
      - wss_curves.png
      - ch_curves.png
      - silhouette_curves.png
      - gap_curves.png
    Also saves per-method CSVs in <output_dir>:
      - metrics_<method>.csv  (k, wss, ch, sil, gap)

    Returns:
      dict of file paths.
    """
    # Map legacy 'scaled_path' -> 'data_path' if provided
    if data_path is None and scaled_path is not None:
        data_path = scaled_path

    _setup_logging(verbosity)
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(random_state)

    # LOAD FEATURE MATRIX (unscaled preferred)
    X = _load_feature_matrix(output_dir, data_path)

    # Basic sanity checks
    if not np.isfinite(X).all():
        logging.warning("Input matrix contains non-finite values; filtering rows with NaNs/Inf.")
        good = np.all(np.isfinite(X), axis=1)
        X = X[good]
    # Drop constant columns (they can break CH / Sil for some datasets)
    col_std = X.std(axis=0, ddof=0)
    if np.any(col_std == 0):
        keep = col_std > 0
        dropped = int((~keep).sum())
        logging.warning("Dropping %d constant feature(s).", dropped)
        X = X[:, keep]

    if X.shape[0] < k_max:
        logging.warning("Samples (%d) < k_max (%d). Consider lowering k_max.", X.shape[0], k_max)
    if X.shape[1] < 2:
        logging.warning("Only %d feature(s) — CH/Sil may be unstable.", X.shape[1])

    meth_map = _methods_map(random_state=random_state)
    if methods:
        # Keep only valid ones (ignore typos)
        meth_map = {m: meth_map[m] for m in methods if m in meth_map}

    ks = list(range(k_min, k_max + 1))

    # Reference datasets for Gap statistic: uniform in the bounding box of X
    mins, maxs = X.min(axis=0), X.max(axis=0)
    refs = [np.random.uniform(mins, maxs, size=X.shape) for _ in range(gap_B)]

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"k": ks, "wss": [], "ch": [], "sil": [], "gap": []}
        for name in meth_map
    }

    # Sweep k for each method
    for name, fn in meth_map.items():
        logging.info("Processing %s ...", name)
        for k in ks:
            try:
                labels = fn(X, k)
                uniq = np.unique(labels)
                wss = _inertia_from_labels(X, labels)
                ch = calinski_harabasz_score(X, labels) if len(uniq) > 1 else np.nan
                sil = silhouette_score(X, labels) if len(uniq) > 1 else np.nan
                gap = _gap_statistic(X, fn, k, refs)
            except Exception as e:
                logging.warning("%s failed at k=%d: %s", name, k, e)
                wss = ch = sil = gap = np.nan

            results[name]["wss"].append(wss)
            results[name]["ch"].append(ch)
            results[name]["sil"].append(sil)
            results[name]["gap"].append(gap)

        # Save per-method CSV
        dfm = pd.DataFrame(results[name])
        csv_path = os.path.join(output_dir, f"metrics_{name}.csv")
        dfm.to_csv(csv_path, index=False)
        results[name]["csv_path"] = csv_path

    # ---- Save one figure PER METRIC ----
    paths: Dict[str, str] = {}
    paths["wss_png"] = _plot_metric_curves(
        ks, results, metric_key="wss",
        ylabel="WSS (Inertia) ↓",
        out_path=os.path.join(figures_dir, "wss_curves.png"),
        logy=True
    )
    paths["ch_png"] = _plot_metric_curves(
        ks, results, metric_key="ch",
        ylabel="Calinski–Harabasz ↑",
        out_path=os.path.join(figures_dir, "ch_curves.png"),
        logy=False
    )
    paths["silhouette_png"] = _plot_metric_curves(
        ks, results, metric_key="sil",
        ylabel="Silhouette ↑",
        out_path=os.path.join(figures_dir, "silhouette_curves.png"),
        logy=False
    )
    paths["gap_png"] = _plot_metric_curves(
        ks, results, metric_key="gap",
        ylabel="Gap Statistic ↑",
        out_path=os.path.join(figures_dir, "gap_curves.png"),
        logy=False
    )

    # include CSV paths in the return payload
    for m in meth_map:
        paths[f"{m}_csv"] = results[m]["csv_path"]

    return paths


# -----------------------
# Optional CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate clustering metrics plots over k (one PNG per metric) using SCALED features by default.")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--data-path", default=None,
                    help="Explicit path to a .npy/.csv/.parquet feature matrix "
                         "(overrides autodetection).")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of methods (KMeans Agglomerative Birch GMM)")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--gap-B", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="count", default=1)
    args = ap.parse_args()

    out = run(
        base_dir=None,
        output_dir=args.output_dir,
        data_path=args.data_path,
        methods=args.methods,
        k_min=args.k_min,
        k_max=args.k_max,
        gap_B=args.gap_B,
        verbosity=args.verbose,
        random_state=args.random_state,
    )
    print("Saved:", out)
