# annotations.py
"""
Interactive annotations for 2D latent scatter plots (matplotlib).

Features
--------
- Click-to-annotate: click a point to show an info box (e.g., mukey, cluster, coords).
- Toggle behavior: clicking a new point moves the annotation; clicking empty space hides it.
- Optional highlight circle to emphasize the selected point.

Typical Usage
-------------
fig, ax = plt.subplots()
sc = ax.scatter(z_mean[:,0], z_mean[:,1], s=12, alpha=0.7)
from annotations import enable_point_annotations
disconnect = enable_point_annotations(
    ax=ax,
    scatter=sc,
    payloads=[
        {"mukey": int(df.loc[i, "mukey"]),
         "cluster": int(df.loc[i, cluster_col]),
         "x": float(z_mean[i,0]), "y": float(z_mean[i,1])}
        for i in range(len(df))
    ],
    fmt=lambda p: f"mukey: {p['mukey']}\ncluster: {p['cluster']}\n({p['x']:.3f}, {p['y']:.3f})",
)
# call disconnect() later to remove the event handlers
__author__ = "Dipal Shah"
__email__  = "dipalshah@missouri.edu"
__license__ = "MIT"
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.patches import Circle


def _default_fmt(payload: Dict) -> str:
    """Fallback annotation text formatter."""
    keys = ["id", "mukey", "cluster", "x", "y"]
    present = [k for k in keys if k in payload]
    if present:
        lines = [f"{k}: {payload[k]}" for k in present]
    else:
        # print everything if no common keys
        lines = [f"{k}: {v}" for k, v in payload.items()]
    return "\n".join(lines)


def enable_point_annotations(
    ax: Axes,
    scatter: PathCollection,
    payloads: Sequence[Dict],
    *,
    fmt: Optional[Callable[[Dict], str]] = None,
    annotation_kwargs: Optional[Dict] = None,
    highlight: bool = True,
    highlight_kwargs: Optional[Dict] = None,
    pickradius: float = 5.0,
) -> Callable[[], None]:
    """
    Attach click-to-annotate behavior to a matplotlib scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the scatter.
    scatter : matplotlib.collections.PathCollection
        The scatter returned by plt.scatter(...).
    payloads : Sequence[Dict]
        One dict per point (same order as the scatter data). Keys can include
        'mukey', 'cluster', 'id', 'x', 'y', etc. Anything is supported;
        the formatter chooses what to show.
    fmt : Callable[[Dict], str], optional
        Function that converts a payload dict to a text string. If None, uses a sensible default.
    annotation_kwargs : dict, optional
        Styling for the annotation box (e.g., bbox dict, fontsize).
    highlight : bool
        Whether to draw a circular marker around the selected point.
    highlight_kwargs : dict, optional
        Styling for the highlight circle (edgecolor, linewidth, alpha, etc.).
    pickradius : float
        Pixel radius for picking points in the scatter.

    Returns
    -------
    disconnect : Callable[[], None]
        Call this function to remove the interactive event handlers.
    """
    if fmt is None:
        fmt = _default_fmt

    # Ensure scatter is pickable
    try:
        scatter.set_picker(True)
    except Exception:
        pass
    try:
        scatter.set_pickradius(pickradius)  # works on some backends
    except Exception:
        pass

    # Prepare a single reusable annotation artist
    if annotation_kwargs is None:
        annotation_kwargs = {}
    bbox_default = dict(boxstyle="round", fc="w", ec="0.3", alpha=0.9, lw=0.8)
    bbox = annotation_kwargs.pop("bbox", bbox_default)

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=bbox,
        **annotation_kwargs,
    )
    ann.set_visible(False)

    # Optional highlight circle to emphasize the selected point
    hl: Optional[Circle] = None
    if highlight:
        if highlight_kwargs is None:
            highlight_kwargs = {}
        hl = Circle((0, 0), radius=0.0, fill=False, **({"ec": "k", "lw": 1.2, "alpha": 0.9} | highlight_kwargs))
        hl.set_visible(False)
        ax.add_patch(hl)

    # Cache point coordinates in data space for fast lookups
    # This respects any transforms already applied by scatter.
    offsets = scatter.get_offsets()  # Nx2 array (x,y)
    if len(offsets) != len(payloads):
        # In degenerate cases (e.g., masked points), try to be forgiving
        n = min(len(offsets), len(payloads))
        offsets = offsets[:n]
        payloads = payloads[:n]

    state = {"visible": False, "ind": None}

    def _show_annotation(ind: int):
        x, y = offsets[ind]
        payload = payloads[ind]
        ann.xy = (x, y)
        ann.set_text(fmt(payload))
        ann.set_visible(True)

        if hl is not None:
            # Circle radius scales with axes span for a consistent feel
            xr = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1.0
            yr = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
            r = 0.01 * max(xr, yr)
            hl.center = (x, y)
            hl.set_radius(r)
            hl.set_visible(True)

        state["visible"] = True
        state["ind"] = ind
        ax.figure.canvas.draw_idle()

    def _hide_annotation():
        ann.set_visible(False)
        if hl is not None:
            hl.set_visible(False)
        state["visible"] = False
        state["ind"] = None
        ax.figure.canvas.draw_idle()

    def on_pick(event):
        if event.artist is not scatter:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return

        # If multiple points are within pickradius, choose the first
        ind = int(event.ind[0])

        # Toggle if the same point
        if state["visible"] and state["ind"] == ind:
            _hide_annotation()
        else:
            _show_annotation(ind)

    def on_button_press(event):
        # Left-click in empty space hides the annotation
        if event.inaxes != ax:
            return
        # If user clicks but not on a pickable point, hide
        contains, _ = scatter.contains(event)
        if not contains and state["visible"]:
            _hide_annotation()

    cid_pick = ax.figure.canvas.mpl_connect("pick_event", on_pick)
    cid_click = ax.figure.canvas.mpl_connect("button_press_event", on_button_press)

    def disconnect():
        """Remove event handlers and artists cleanly."""
        try:
            ax.figure.canvas.mpl_disconnect(cid_pick)
        except Exception:
            pass
        try:
            ax.figure.canvas.mpl_disconnect(cid_click)
        except Exception:
            pass
        try:
            ann.remove()
        except Exception:
            pass
        if hl is not None:
            try:
                hl.remove()
            except Exception:
                pass
        ax.figure.canvas.draw_idle()

    return disconnect


# Convenience wrapper for common latent-plot case
def attach_latent_annotations(
    ax: Axes,
    scatter: PathCollection,
    z_mean,
    *,
    ids: Optional[Sequence] = None,
    clusters: Optional[Sequence[int]] = None,
    id_key: str = "mukey",
    cluster_key: str = "cluster",
    fmt: Optional[Callable[[Dict], str]] = None,
    **kwargs,
) -> Callable[[], None]:
    """
    Build payloads from z_mean (+ optional ids/clusters) and enable annotations.

    Parameters
    ----------
    ax, scatter : see enable_point_annotations(...)
    z_mean : array-like of shape (n_samples, 2)
    ids : optional identifiers per point (e.g., mukey)
    clusters : optional cluster labels per point
    id_key, cluster_key : keys used in the payload shown in the annotation
    fmt : optional formatter; defaults to: "{id_key}: <id>\\n{cluster_key}: <cl>\\n(x,y)"

    Returns
    -------
    disconnect : Callable[[], None]
    """
    import numpy as np

    z = np.asarray(z_mean)
    n = len(z)
    if ids is None:
        ids = list(range(n))
    if clusters is None:
        clusters = [-1] * n

    if fmt is None:
        def _fmt(p: Dict) -> str:
            return f"{id_key}: {p.get(id_key)}\n{cluster_key}: {p.get(cluster_key)}\n({p['x']:.3f}, {p['y']:.3f})"
        fmt = _fmt

    payloads = [
        {id_key: ids[i], cluster_key: clusters[i], "x": float(z[i, 0]), "y": float(z[i, 1])}
        for i in range(n)
    ]

    return enable_point_annotations(ax, scatter, payloads, fmt=fmt, **kwargs)
