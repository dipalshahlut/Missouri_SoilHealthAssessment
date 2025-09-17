# clustering_evaluation.py
"""
Evaluate clustering algorithms across a k-range, pick best-k per method,
and attach final labels back to your dataframe.

Public API:
  - evaluate_multiple_algorithms(X, methods, k_range, random_state=42)
  - get_best_clustering_results(df, z_mean, best_k_for, methods, random_state=42)
  - fit_predict_labels(method, k, X, random_state=42)   # compute labels on demand

Supported methods (case-insensitive):
  - "KMeans", "Agglomerative", "Birch", "GMM"

Usage:
1. Evaluate all methods (default k=2–20):
python clustering_evaluation.py \
  --z-path /path/to/data/aggResult/z_mean.npy \
  --methods KMeans Agglomerative Birch GMM \
   --k-min 10 \
  --k-max 20 \
  --output /path/to/data/aggResult/eval_scores.csv \
  --json-out /path/to/data/aggResult/eval_scores.json

__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ----------------------- method canonicalization -----------------------------

_CANON = {
    "kmeans": "KMeans",
    "agglomerative": "Agglomerative",
    "birch": "Birch",
    "gmm": "GMM",
}

def _canon(method: str) -> str:
    m = method.strip().lower()
    if m not in _CANON:
        raise ValueError(f"Unsupported method '{method}'. Supported: {list(_CANON.values())}")
    return _CANON[m]

# ----------------------- core fit/predict -----------------------------------

def _fit_predict_labels(method: str, k: int, X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Fit a clustering model for a given method/k and return labels.
    Internal helper; use the public wrapper fit_predict_labels() in other modules.
    """
    X = np.asarray(X)
    m = _canon(method)

    if m == "KMeans":
        # scikit-learn>=1.4 supports n_init="auto"; if you hit an error, set n_init=10
        model = KMeans(n_clusters=int(k), n_init="auto", random_state=random_state)
        return model.fit_predict(X)

    if m == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        return model.fit_predict(X)

    if m == "Birch":
        model = Birch(n_clusters=int(k))
        return model.fit_predict(X)

    if m == "GMM":
        model = GaussianMixture(
            n_components=int(k),
            #covariance_type="full",
            #n_init=5,
            random_state=random_state,
        )
        return model.fit(X).predict(X)

    # Should never happen due to _canon()
    raise ValueError(f"Unsupported method: {method}")

# Public, stable name to import from other modules
def fit_predict_labels(method: str, k: int, X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Public wrapper to compute labels for (method, k)."""
    return _fit_predict_labels(method, int(k), X, random_state=random_state)

# ----------------------- metrics & evaluation --------------------------------

def _safe_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering metrics with guards for degenerate cases."""
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan

    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan

    return {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)}

def evaluate_multiple_algorithms(
    X: np.ndarray,
    methods: Iterable[str],
    k_range: Iterable[int],
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, int], Dict[str, float]]:
    """
    Evaluate each method over k_range and compute metrics.

    Returns:
      scores: dict s.t. scores[method][metric][k] = value
      best_k_for: best k per method (primary=Silhouette, tie1=CH, tie2=DB (lower better))
      best_score_for: best Silhouette per method
    """
    X = np.asarray(X)
    methods = [_canon(m) for m in methods]
    scores: Dict[str, Dict[str, Dict[int, float]]] = {}
    best_k_for: Dict[str, int] = {}
    best_score_for: Dict[str, float] = {}

    for m in methods:
        scores[m] = {"silhouette": {}, "calinski_harabasz": {}, "davies_bouldin": {}}

        for k in k_range:
            k = int(k)
            if k < 2 or k > len(X):
                continue
            try:
                labels = _fit_predict_labels(m, k, X, random_state=random_state)
            except Exception as e:
                logging.warning("Method %s failed at k=%s: %s", m, k, e)
                scores[m]["silhouette"][k] = np.nan
                scores[m]["calinski_harabasz"][k] = np.nan
                scores[m]["davies_bouldin"][k] = np.nan
                continue

            metr = _safe_metrics(X, labels)
            scores[m]["silhouette"][k] = metr["silhouette"]
            scores[m]["calinski_harabasz"][k] = metr["calinski_harabasz"]
            scores[m]["davies_bouldin"][k] = metr["davies_bouldin"]

        ks = list(scores[m]["silhouette"].keys())
        if not ks:
            logging.warning("No valid scores for method %s; skipping selection.", m)
            continue

        def _score_tuple(kk: int):
            sil = scores[m]["silhouette"].get(kk, np.nan)
            ch = scores[m]["calinski_harabasz"].get(kk, np.nan)
            db = scores[m]["davies_bouldin"].get(kk, np.nan)
            sil_s = -np.inf if np.isnan(sil) else sil
            ch_s = -np.inf if np.isnan(ch) else ch
            db_s = -np.inf if np.isnan(db) else -db  # lower DB is better
            return (sil_s, ch_s, db_s)

        best_k = max(ks, key=_score_tuple)
        best_k_for[m] = int(best_k)
        best_score_for[m] = float(scores[m]["silhouette"].get(int(best_k), np.nan))

    return scores, best_k_for, best_score_for

def get_best_clustering_results(
    df: pd.DataFrame,
    z_mean: np.ndarray,
    best_k_for: Dict[str, int],
    methods: Iterable[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit each method at its selected best k and append a label column:
      f"{Method}_best{best_k}"
    Returns a NEW dataframe with added columns.
    """
    X = np.asarray(z_mean)
    out = df.copy()
    for m in [_canon(mm) for mm in methods]:
        k = int(best_k_for.get(m, 0))
        if k < 2:
            logging.info("No valid best k for %s; skipping.", m)
            continue
        try:
            labels = _fit_predict_labels(m, k, X, random_state=random_state)
        except Exception as e:
            logging.warning("Could not fit %s at k=%d for final labels: %s", m, k, e)
            continue
        out[f"{m}_best{k}"] = labels
    return out

__all__ = [
    "evaluate_multiple_algorithms",
    "get_best_clustering_results",
    "fit_predict_labels",
]

def _build_argparser():
    p = argparse.ArgumentParser(description="Evaluate clustering algorithms on latent space (z_mean).")
    p.add_argument("--z-path", required=True, help="Path to z_mean.npy")
    p.add_argument("--methods", nargs="+", required=True,
                   choices=["KMeans", "Agglomerative", "Birch", "GMM"],
                   help="Clustering methods to evaluate")
    p.add_argument("--k-min", type=int, default=2, help="Minimum k (default: 2)")
    p.add_argument("--k-max", type=int, default=15, help="Maximum k (default: 15)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", default=None, help="Optional CSV to save tidy scores")
    p.add_argument("--json-out", default=None, help="Optional JSON file to save best_k_for and best_score_for")
    return p

def main(argv=None):
    args = _build_argparser().parse_args(argv)
    X = np.load(args.z_path)

    # Evaluate
    scores, best_k_for, best_score_for = evaluate_multiple_algorithms(
        X, methods=args.methods, k_range=range(args.k_min, args.k_max + 1),
        random_state=args.random_state
    )

    # Convert scores to tidy DataFrame
    records = []
    for m, metr_dict in scores.items():
        for metric, kv in metr_dict.items():
            for k, v in kv.items():
                records.append({"method": m, "metric": metric, "k": k, "value": v})
    tidy = pd.DataFrame.from_records(records)

    print("\n=== Best k per method ===")
    for m, k in best_k_for.items():
        print(f"{m}: k={k}, silhouette={best_score_for[m]:.3f}")

    # Save CSV if requested
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(args.output, index=False)
        print(f"\nScores saved to {args.output}")

    # Save JSON if requested
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"best_k_for": best_k_for, "best_score_for": best_score_for}, f, indent=2)
        print(f"Best results saved to {args.json_out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
