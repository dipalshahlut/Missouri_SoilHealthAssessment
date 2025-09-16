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
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

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
    # keeping the private symbol available if some modules still import it
    "_fit_predict_labels",
]
