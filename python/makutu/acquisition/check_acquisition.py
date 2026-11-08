"""
check_acquisition.py — Validate and display acquisition geometry from SEG-Y files.

Reads all SEG-Y files in a directory (or a list of files), extracts source and
receiver coordinates from the trace headers, and reports:

  - Number of shots and traces
  - Per-shot receiver count (uniformity check)
  - Spatial coverage: source and receiver bounding boxes
  - Offset statistics (min / max / mean)
  - Fold map (number of traces per receiver bin)

A geometry map is saved (or displayed) showing source and receiver positions.

Usage
-----
    # check all .segy files in a directory
    python apps/acquisition/check_acquisition.py --dir acquisition --out survey

    # check specific files
    python apps/acquisition/check_acquisition.py shot_00001.segy shot_00002.segy

    # read word positions matching create_equispaced_segy.py defaults
    python apps/acquisition/check_acquisition.py --dir acquisition --word-sx 22 --word-sy 23
"""

import os
import struct

import numpy as np

from .decision_create import read_segy_geometry


# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_segy_geometry(path):
    """
    Source (first trace) and receiver coordinates [m] of a SEG-Y file, from the standard
    trace-header words (see :func:`makutu.acquisition.decision_create.read_segy_geometry`).

    Returns
    -------
    src : tuple (sx, sy, sz)
    rcv : np.ndarray shape (n_traces, 3)
    n_samples : int
    dt_us : int   sample interval [µs]
    """
    src, rcv, n_samples, dt_us = read_segy_geometry(path)
    return tuple(src[0]), rcv, n_samples, dt_us


# ─────────────────────────────────────────────────────────────────────────────
# Geometry checks
# ─────────────────────────────────────────────────────────────────────────────

def check_acquisition(paths):
    """
    Read geometry from a list of SEG-Y files (one per shot) and return a summary dict.

    Returns
    -------
    summary : dict with keys:
        n_shots, n_traces_per_shot, n_samples, dt_us,
        src_positions  (n_shots × 3),
        rcv_positions  (n_rcv_per_shot × 3),
        offsets        (n_shots × n_rcv_per_shot),
        non_uniform_shots  list of shot indices with wrong receiver count.
    """
    if not paths:
        raise ValueError("No SEG-Y files supplied.")

    shots = []
    n_rcv_expected = None
    non_uniform = []
    n_samples_ref = None
    dt_us_ref = None

    for i, p in enumerate(paths):
        src, rcv, n_samples, dt_us = _read_segy_geometry(p)

        if n_samples_ref is None:
            n_samples_ref = n_samples
            dt_us_ref = dt_us
        if n_rcv_expected is None:
            n_rcv_expected = len(rcv)
        elif len(rcv) != n_rcv_expected:
            non_uniform.append(i)

        shots.append({"src": src, "rcv": rcv, "file": os.path.basename(p)})

    src_arr = np.array([s["src"] for s in shots], dtype=np.float64)

    # Use the receiver array from the first shot (reference)
    rcv_ref = shots[0]["rcv"]

    # Offsets: distance from each source to each receiver
    offsets = np.zeros((len(shots), len(rcv_ref)))
    for i, s in enumerate(shots):
        dx = s["rcv"][:, 0] - s["src"][0]
        dy = s["rcv"][:, 1] - s["src"][1]
        offsets[i] = np.sqrt(dx**2 + dy**2)

    return {
        "n_shots":            len(shots),
        "n_traces_per_shot":  n_rcv_expected,
        "n_samples":          n_samples_ref,
        "dt_us":              dt_us_ref,
        "src_positions":      src_arr,
        "rcv_positions":      rcv_ref,
        "offsets":            offsets,
        "non_uniform_shots":  non_uniform,
        "shots":              shots,
    }


def print_report(summary):
    """Print a text summary of the acquisition geometry."""
    s = summary
    print("=" * 60)
    print("Acquisition geometry report")
    print("=" * 60)
    print(f"  Shots             : {s['n_shots']}")
    print(f"  Receivers/shot    : {s['n_traces_per_shot']}")
    print(f"  Total traces      : {s['n_shots'] * s['n_traces_per_shot']}")
    if s["n_samples"] is not None:
        dt_ms   = s["dt_us"] / 1000.0
        tmax_ms = s["n_samples"] * dt_ms
        print(f"  Samples/trace     : {s['n_samples']}")
        print(f"  Sample interval   : {dt_ms:.3f} ms  ({s['dt_us']} µs)")
        print(f"  Record length     : {tmax_ms:.1f} ms")
    else:
        print(f"  Samples/trace     : n/a (decision file only)")
        print(f"  Sample interval   : n/a")

    if s["non_uniform_shots"]:
        print(f"\n  WARNING: {len(s['non_uniform_shots'])} shots have "
              f"non-uniform receiver count: {s['non_uniform_shots']}")
    else:
        print("  Receiver count    : uniform (OK)")

    src = s["src_positions"]
    rcv = np.array(s["rcv_positions"])
    print(f"\n  Source extent:")
    print(f"    X : [{src[:,0].min():.1f}, {src[:,0].max():.1f}]  "
          f"(dx = {np.diff(np.unique(src[:,0])).mean():.1f} m)" if len(np.unique(src[:,0]))>1 else
          f"    X : [{src[:,0].min():.1f}, {src[:,0].max():.1f}]")
    print(f"    Y : [{src[:,1].min():.1f}, {src[:,1].max():.1f}]")
    print(f"    Z : [{src[:,2].min():.1f}, {src[:,2].max():.1f}]")

    print(f"\n  Receiver extent (shot 1):")
    print(f"    X : [{rcv[:,0].min():.1f}, {rcv[:,0].max():.1f}]")
    print(f"    Y : [{rcv[:,1].min():.1f}, {rcv[:,1].max():.1f}]")
    print(f"    Z : [{rcv[:,2].min():.1f}, {rcv[:,2].max():.1f}]")

    off = s["offsets"]
    print(f"\n  Offsets [m]:")
    print(f"    min  : {off.min():.1f}")
    print(f"    max  : {off.max():.1f}")
    print(f"    mean : {off.mean():.1f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_acquisition(summary, out_path=None, show=True):
    """
    Plot acquisition map (sources + receivers) and offset-fold histogram.

    Parameters
    ----------
    summary : dict   As returned by :func:`check_acquisition`.
    out_path : str   If provided, save the figure to this path.
    show : bool      Call plt.show() after plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    src = summary["src_positions"]
    rcv = np.array(summary["rcv_positions"])
    offsets = summary["offsets"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 1. Acquisition map ───────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(rcv[:, 0], rcv[:, 1], s=6, c="steelblue",
               label=f"Receivers ({len(rcv)})", zorder=2, alpha=0.6)
    ax.scatter(src[:, 0], src[:, 1], s=30, c="crimson", marker="*",
               label=f"Sources ({len(src)})", zorder=3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Acquisition map")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)

    # ── 2. Offset distribution ───────────────────────────────────────────────
    ax = axes[1]
    ax.hist(offsets.ravel(), bins=50, color="steelblue", edgecolor="white", lw=0.3)
    ax.set_xlabel("Offset [m]")
    ax.set_ylabel("Count")
    ax.set_title("Offset distribution")
    ax.grid(True, lw=0.3, axis="y")

    # ── 3. Fold map (receiver bins coloured by hit count) ────────────────────
    ax = axes[2]
    # Count how many shots cover each receiver
    n_shots = summary["n_shots"]
    sc = ax.scatter(rcv[:, 0], rcv[:, 1], c=[n_shots] * len(rcv),
                    s=8, cmap="viridis", vmin=0, vmax=n_shots)
    ax.scatter(src[:, 0], src[:, 1], s=20, c="crimson", marker="*", zorder=3)
    fig.colorbar(sc, ax=ax, label="Fold (shots per receiver)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Fold map")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Geometry plot → {out_path}")
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Decision-file reader path
# ─────────────────────────────────────────────────────────────────────────────

def check_from_decision(rootname,
                        w_sx=22, w_sy=23, w_sz=14,
                        w_rx=24, w_ry=25, w_rz=13):
    """
    Build a summary dict by reading geometry from decision files.

    Much faster than scanning individual SEG-Y files for large surveys.

    Parameters
    ----------
    rootname : str
        Decision-file rootname (e.g. ``"acquisition/survey"``).
        Reads ``<rootname>.mapped.data`` and ``<rootname>.mapped.list``.
    w_sx … w_rz : int
        Word numbers for source / receiver coordinates.

    Returns
    -------
    summary : dict   Compatible with :func:`print_report` and
                     :func:`plot_acquisition`.
    """
    from makutu.acquisition.decision_reader import read_decision_file

    print(f"Reading decision file: {rootname}.mapped.data ...")
    d_src, d_rcv, n_shots = read_decision_file(
        rootname,
        swords=[w_sx, w_sy, w_sz],
        rwords=[w_rx, w_ry, w_rz],
        scalco=1, scalel=1,          # no scaling — coordinates stored as-is
    )

    # Convert to arrays
    src_arr = np.array([d_src[f"src{i+1}"] for i in range(n_shots)],
                       dtype=np.float64)

    # Receiver set from shot 1 (reference)
    rcv_ref = np.array(d_rcv["rcv_1"], dtype=np.float64)

    # Check uniformity across all shots
    n_rcv_ref = len(rcv_ref)
    non_uniform = [i for i in range(n_shots)
                   if len(d_rcv.get(f"rcv_{i+1}", [])) != n_rcv_ref]

    # Offsets for all shots
    offsets = np.zeros((n_shots, n_rcv_ref))
    for i in range(n_shots):
        rcv_i = np.array(d_rcv.get(f"rcv_{i+1}", rcv_ref), dtype=np.float64)
        n = min(len(rcv_i), n_rcv_ref)
        dx = rcv_i[:n, 0] - src_arr[i, 0]
        dy = rcv_i[:n, 1] - src_arr[i, 1]
        offsets[i, :n] = np.sqrt(dx**2 + dy**2)

    # n_samples / dt not available from decision files alone
    return {
        "n_shots":            n_shots,
        "n_traces_per_shot":  n_rcv_ref,
        "n_samples":          None,
        "dt_us":              None,
        "src_positions":      src_arr,
        "rcv_positions":      rcv_ref,
        "offsets":            offsets,
        "non_uniform_shots":  non_uniform,
        "shots":              [{"src": d_src[f"src{i+1}"],
                                "rcv": d_rcv.get(f"rcv_{i+1}", [])}
                               for i in range(n_shots)],
    }
