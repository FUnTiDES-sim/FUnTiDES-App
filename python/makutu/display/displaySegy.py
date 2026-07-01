"""
displaySegy.py — Display SEG-Y shot gathers.

Reads one or more SEG-Y files and plots each as a seismic image
(imshow: time on the vertical axis, trace index on the horizontal).
Falls back to a pure-Python reader if segyio is not installed.

Usage
-----
    python displaySegy.py shot_001.segy shot_002.segy \\
        [--clip 98] [--cmap seismic] [--gain 1.0] \\
        [--dt 2.0] [--tmax 3000] \\
        [--cols 3] [--output gathers.png] [--no-show]
"""

import argparse
import os
import struct
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SEG-Y readers — segyio preferred, pure-Python fallback
# ---------------------------------------------------------------------------

def _read_segy_segyio(path):
    """Return (data, dt_us) using segyio. data shape: (n_samples, n_traces)."""
    import segyio
    with segyio.open(path, "r+", ignore_geometry=True) as f:
        data  = np.stack([f.trace[i] for i in range(len(f.trace))], axis=1)
        dt_us = segyio.tools.dt(f)   # µs
    return data.astype(np.float32), float(dt_us)


def _read_segy_pure(path):
    """
    Return (data, dt_us) using a standalone pure-Python SEG-Y reader.

    Handles both standard-compliant files (dt @ offset 16, nt @ offset 20)
    and older files written with the swapped convention (nt @ offset 16,
    dt @ offset 18).  Falls back to the first trace header when the binary
    reel header yields nt = 0.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    bh = raw[3200:3600]

    dt_us = float(struct.unpack(">H", bh[16:18])[0])   # standard: dt in µs
    nt    = struct.unpack(">H", bh[20:22])[0]            # standard: n_samples
    fmt   = struct.unpack(">H", bh[24:26])[0]            # data format code

    # Fallback 1: nt=0 → try first trace header (bytes 115-116, offset 114)
    if nt == 0 and len(raw) > 3600 + 118:
        nt    = struct.unpack(">H", raw[3600 + 114: 3600 + 116])[0]
        dt_th = struct.unpack(">H", raw[3600 + 116: 3600 + 118])[0]
        if dt_th > 0:
            dt_us = float(dt_th)

    # Fallback 2: nt still 0, binary header may have swapped n_samples↔dt
    # (old writer bug: n_samples at offset 16, dt at offset 18)
    if nt == 0:
        nt_candidate = struct.unpack(">H", bh[16:18])[0]
        dt_candidate = struct.unpack(">H", bh[18:20])[0]
        if nt_candidate > 0 and dt_candidate > 0:
            nt    = nt_candidate
            dt_us = float(dt_candidate)

    if nt == 0:
        raise ValueError(f"Cannot determine sample count from {path}")

    trace_size = 240 + nt * 4
    n_trace    = (len(raw) - 3600) // trace_size

    if n_trace <= 0:
        raise ValueError(f"No complete traces found in {path}")

    data = np.zeros((nt, n_trace), dtype=np.float32)
    for i in range(n_trace):
        off = 3600 + i * trace_size + 240
        sd  = raw[off: off + nt * 4]
        if fmt == 1:
            # IBM float conversion
            vals = np.array([
                struct.unpack_from(">I", sd, j * 4)[0] for j in range(nt)
            ], dtype=np.uint32)
            ieee = np.zeros(nt, dtype=np.float32)
            nz   = vals != 0
            sign = np.where(vals >> 31, -1.0, 1.0)
            exp  = ((vals >> 24) & 0x7F).astype(np.float64) - 64.0
            mant = (vals & 0x00FFFFFF).astype(np.float64) / 16777216.0
            ieee[nz] = (sign[nz] * mant[nz] * 16.0 ** exp[nz]).astype(np.float32)
            data[:, i] = ieee
        else:
            data[:, i] = np.frombuffer(sd, dtype=">f4")

    return data, dt_us


def read_segy(path):
    """Read a SEG-Y file; try segyio first, fall back to pure-Python."""
    try:
        return _read_segy_segyio(path)
    except Exception:
        return _read_segy_pure(path)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _clip_scale(data, clip_pct, gain):
    """Return (data_scaled, vmax) after gain and percentile clip."""
    d = data * gain
    vmax = float(np.percentile(np.abs(d), clip_pct))
    if vmax == 0.0:
        vmax = 1.0
    return d, vmax


def plot_gather(ax, data, dt_us, title="", clip_pct=98, gain=1.0,
                cmap="seismic", tmax_ms=None):
    """
    Plot one shot gather on *ax*.

    Parameters
    ----------
    data    : (n_samples, n_traces) float32
    dt_us   : float   Sample interval [µs]
    tmax_ms : float   Clip display to this time [ms]; None = show all
    """
    d, vmax = _clip_scale(data, clip_pct, gain)

    dt_ms    = dt_us / 1000.0
    t_end_ms = d.shape[0] * dt_ms

    if tmax_ms is not None and tmax_ms < t_end_ms:
        n_show = max(1, int(tmax_ms / dt_ms))
        d      = d[:n_show, :]
        t_end_ms = n_show * dt_ms

    ax.imshow(
        d,
        aspect="auto",
        origin="upper",
        extent=[0.5, d.shape[1] + 0.5, t_end_ms, 0.0],
        vmin=-vmax, vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Trace", fontsize=9)
    ax.set_ylabel("Time [ms]", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+",
                   help="SEG-Y file(s) to display")
    p.add_argument("--clip", type=float, default=98.0,
                   help="Percentile clip for colour scale (default: 98)")
    p.add_argument("--cmap", default="seismic",
                   help="Matplotlib colormap (default: seismic)")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Amplitude gain factor (default: 1.0)")
    p.add_argument("--dt", type=float, default=None,
                   help="Override sample interval [ms] (default: read from header)")
    p.add_argument("--tmax", type=float, default=None,
                   help="Display up to this time [ms] (default: full record)")
    p.add_argument("--cols", type=int, default=None,
                   help="Number of subplot columns (default: auto)")
    p.add_argument("--output", default=None,
                   help="Save figure to this file instead of (or in addition to) showing")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive window (useful with --output)")
    return p.parse_args()


def main():
    args  = parse_args()
    files = args.files

    n = len(files)
    cols = args.cols if args.cols else min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 5 * rows),
        squeeze=False,
    )

    for idx, path in enumerate(files):
        ax = axes[idx // cols][idx % cols]

        if not os.path.isfile(path):
            ax.set_visible(False)
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        print(f"Reading {path} ...", end=" ", flush=True)
        try:
            data, dt_us = read_segy(path)
        except Exception as exc:
            ax.set_visible(False)
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        if args.dt is not None:
            dt_us = args.dt * 1000.0   # ms → µs

        n_s, n_t = data.shape
        dt_ms    = dt_us / 1000.0
        print(f"{n_t} traces  {n_s} samples  dt={dt_ms:.3f} ms")

        plot_gather(
            ax, data, dt_us,
            title=os.path.basename(path),
            clip_pct=args.clip,
            gain=args.gain,
            cmap=args.cmap,
            tmax_ms=args.tmax,
        )

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
