"""
create_equispaced_segy.py — Build a regular 2D/3D equispaced acquisition and
write one SEG-Y placeholder file per shot.

The SEG-Y files contain zero-amplitude traces with all geometry headers
correctly populated.  They are meant to be filled with simulation data later.
Decision files (mapped + index) ready for use with decision_create are also
written.

Geometry model
--------------
Sources are distributed on a regular 2D grid inside
  [xmin + margin, xmax - margin]  ×  [ymin + margin, ymax - margin]
at depth *z_src* (positive = below surface).

Receivers span the full aperture:
  [xmin + margin, xmax - margin]  ×  [ymin + margin, ymax - margin]
at depth *z_rcv*, sampled independently with --n-rcv-x / --n-rcv-y.
The same receiver grid is shared by all shots.

For a pure 2D survey set --n-src-y 1 --n-rcv-y 1: sources and receivers
are placed at y = (ymin + ymax) / 2.

SEG-Y convention
----------------
Geometry in the standard trace-header words (sx, sy, sdepth, gx, gy, gelev), as integers
in millimetres (scalco = scalel = -1000); depths positive down.
Trace data: IEEE float-32, big-endian (format code 5).
Sample interval stored in microseconds.

Usage
-----
    python apps/acquisition/create_equispaced_segy.py \\
        --xmin 0 --xmax 5000 --ymin 0 --ymax 5000 \\
        --n-src-x 50 --n-src-y 1 \\
        --n-rcv-x 200 --n-rcv-y 1 \\
        --z-src 10 --z-rcv 5 \\
        --n-samples 1001 --dt 2.0 \\
        --out survey --output-dir acquisition

Library usage::

    from makutu.acquisition.create_equispaced_segy import (
        equispaced_shots, write_segy_shot, write_survey,
    )
    shots = equispaced_shots(...)
    write_survey(shots, out="survey", output_dir="acquisition", ...)
"""

import os
import struct

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y writer  (pure Python, no segyio required, format code 5 = IEEE float32)
# ─────────────────────────────────────────────────────────────────────────────

_SEGY_TEXTLINE = 80
_SEGY_TEXTLINES = 40
_SEGY_BINARY_SIZE = 400
_SEGY_TRACE_HEADER_SIZE = 240


def _ebcdic_header(lines):
    """Build a 3200-byte EBCDIC text header from up to 40 ASCII lines."""
    rows = []
    for i in range(_SEGY_TEXTLINES):
        raw = lines[i] if i < len(lines) else ""
        padded = raw[:_SEGY_TEXTLINE].ljust(_SEGY_TEXTLINE)
        rows.append(padded.encode("cp500"))   # cp500 = EBCDIC
    return b"".join(rows)


def _binary_reel_header(n_samples, dt_us):
    """
    Build a minimal 400-byte binary reel header.

    Parameters
    ----------
    n_samples : int   Number of samples per trace.
    dt_us     : int   Sample interval [µs].
    """
    buf = bytearray(_SEGY_BINARY_SIZE)
    # Bytes 17-18 (offset 16): sample interval in µs
    struct.pack_into(">H", buf, 16, min(dt_us,    65535))
    # Bytes 21-22 (offset 20): number of data samples per data trace
    struct.pack_into(">H", buf, 20, min(n_samples, 65535))
    # Bytes 25-26 (offset 24): data sample format code  5 = IEEE float32
    struct.pack_into(">H", buf, 24, 5)
    # Bytes 3501-3502 (offset 300): SEG-Y revision, 0x0100 = rev 1.0
    struct.pack_into(">H", buf, 300, 256)
    # Bytes 3503-3504 (offset 302): fixed-length trace flag (1 = fixed)
    struct.pack_into(">H", buf, 302, 1)
    return bytes(buf)


def _word(buf, n, value):
    """Pack big-endian int32 *value* into word *n* (1-based) of *buf*."""
    struct.pack_into(">i", buf, (n - 1) * 4, int(value))


def _trace_header(trace_seq, field_record, n_samples, dt_us, src, rcv, scalar=-1000):
    """
    Build a 240-byte SEG-Y trace header with the geometry words read by makutu
    (:class:`~makutu.acquisition.SEGYAcquisition`,
    :func:`~makutu.acquisition.decision_create.read_segy_geometry`): source sx, sy (bytes
    73, 77) and depth sdepth (byte 49), receiver gx, gy (bytes 81, 85) and depth gelev
    (byte 41), positive down, scaled by ``scalar`` (scalco and scalel; -1000: millimetres).
    """
    buf = bytearray(_SEGY_TRACE_HEADER_SIZE)
    factor = float(-scalar) if scalar < 0 else 1.0 / scalar

    _word(buf, 1, trace_seq)        # trace sequence number (line)
    _word(buf, 2, trace_seq)        # trace sequence number (file)
    _word(buf, 3, field_record)     # original field record number
    _word(buf, 4, trace_seq)        # trace number within field record

    _word(buf, 11, round(rcv[2] * factor))      # gelev: receiver depth
    _word(buf, 13, round(src[2] * factor))      # sdepth: source depth
    struct.pack_into(">hh", buf, 68, scalar, scalar)   # scalel, scalco
    _word(buf, 19, round(src[0] * factor))      # sx
    _word(buf, 20, round(src[1] * factor))      # sy
    _word(buf, 21, round(rcv[0] * factor))      # gx
    _word(buf, 22, round(rcv[1] * factor))      # gy

    # n_samples (bytes 115-116, int16)  and  sample interval µs (bytes 117-118, int16)
    struct.pack_into(">H", buf, 114, min(n_samples, 65535))
    struct.pack_into(">H", buf, 116, min(dt_us, 65535))

    return bytes(buf)


def write_segy_shot(path, shot, n_samples, dt_us, data=None, field_record=1):
    """
    Write a single-shot SEG-Y file: zero-amplitude traces, or ``data``. The geometry is
    in the standard trace-header words (see :func:`_trace_header`).

    Parameters
    ----------
    path : str
        Output file path.
    shot : dict
        ``{"src": [x, y, z], "rcv": [[x, y, z], ...]}``.
    n_samples : int
        Number of time samples per trace.
    dt_us : int
        Sample interval [µs].
    data : np.ndarray, shape (n_samples, n_receivers), optional
        Trace data (e.g. a modelled seismogram); zeros when omitted.
    field_record : int
        Field record number written in the trace headers (e.g. the shot number).
    """
    sx, sy, sz = [float(v) for v in shot["src"]]
    rcv_list = shot["rcv"]
    n_traces = len(rcv_list)
    if data is not None:
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != n_traces:
            raise ValueError(f"data must have shape (n_samples, {n_traces}), got {data.shape}")
        n_samples = data.shape[0]

    text_lines = [
        f"C01  SEG-Y shot file: {os.path.basename(path)}",
        f"C02  Source  : ({sx:g}, {sy:g}, {sz:g}) m",
        f"C03  Traces  : {n_traces}   Samples: {n_samples}   dt: {dt_us} us",
        "C04  Data format: IEEE float-32",
        "C05  Zero-amplitude placeholder - fill with simulation data" if data is None
        else "C05  Modelled data",
    ]

    zero_data = np.zeros(n_samples, dtype=">f4").tobytes()

    with open(path, "wb") as fh:
        fh.write(_ebcdic_header(text_lines))
        fh.write(_binary_reel_header(n_samples, dt_us))

        for i, rcv in enumerate(rcv_list):
            hdr = _trace_header(i + 1, field_record, n_samples, dt_us, (sx, sy, sz), rcv)
            fh.write(hdr)
            fh.write(zero_data if data is None else data[:, i].astype(">f4").tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builder
# ─────────────────────────────────────────────────────────────────────────────

def equispaced_shots(xmin, xmax, ymin, ymax,
                     n_src_x, n_src_y,
                     n_rcv_x, n_rcv_y,
                     z_src=0.0, z_rcv=0.0,
                     margin=0.0):
    """
    Build a list of shot dicts for a regular equispaced acquisition.

    Sources and receivers are placed on independent 2D grids inside the
    domain [xmin+margin, xmax-margin] × [ymin+margin, ymax-margin].

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float   Domain limits [m].
    n_src_x, n_src_y : int           Source grid size in X and Y.
    n_rcv_x, n_rcv_y : int           Receiver grid size in X and Y.
    z_src, z_rcv : float             Source / receiver depth [m].
    margin : float                   Margin inset from domain edge [m].

    Returns
    -------
    shots : list of dict  ``{"src": [x,y,z], "rcv": [[x,y,z], ...]}``
    """
    x0, x1 = xmin + margin, xmax - margin
    y0, y1 = ymin + margin, ymax - margin

    # Receiver grid (shared by all shots)
    if n_rcv_x == 1:
        rcv_xs = np.array([(x0 + x1) / 2.0])
    else:
        rcv_xs = np.linspace(x0, x1, n_rcv_x)

    if n_rcv_y == 1:
        rcv_ys = np.array([(y0 + y1) / 2.0])
    else:
        rcv_ys = np.linspace(y0, y1, n_rcv_y)

    rcv_grid = [[float(x), float(y), float(z_rcv)]
                for y in rcv_ys for x in rcv_xs]

    # Source grid
    if n_src_x == 1:
        src_xs = np.array([(x0 + x1) / 2.0])
    else:
        src_xs = np.linspace(x0, x1, n_src_x)

    if n_src_y == 1:
        src_ys = np.array([(y0 + y1) / 2.0])
    else:
        src_ys = np.linspace(y0, y1, n_src_y)

    shots = []
    for sy_val in src_ys:
        for sx_val in src_xs:
            shots.append({
                "src": [float(sx_val), float(sy_val), float(z_src)],
                "rcv": list(rcv_grid),
            })

    return shots


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def write_survey(shots, out, output_dir=".",
                 n_samples=1001, dt_ms=2.0,
                 w_sx=22, w_sy=23, w_sz=14,
                 w_rx=24, w_ry=25, w_rz=13):
    """
    Write one SEG-Y file per shot and the corresponding decision files.

    Parameters
    ----------
    shots : list of dict
        As returned by :func:`equispaced_shots`.
    out : str
        Rootname for decision files and SEG-Y filenames.
    output_dir : str
        Directory for all output files.
    n_samples : int
        Samples per trace.
    dt_ms : float
        Sample interval [ms].
    w_sx, …, w_rz : int
        Decision-file word numbers of the source / receiver coordinates (the SEG-Y files
        use the standard trace-header words, see :func:`write_segy_shot`).
    """
    # Lazy import to avoid circular dependency when used as a library
    from makutu.acquisition.decision_create import create_decision_files

    os.makedirs(output_dir, exist_ok=True)
    dt_us = int(round(dt_ms * 1000.0))

    segy_files = []
    for i, shot in enumerate(shots):
        fname = f"{out}_shot_{i + 1:05d}.segy"
        fpath = os.path.join(output_dir, fname)
        write_segy_shot(fpath, shot, n_samples, dt_us, field_record=i + 1)
        segy_files.append(fname)   # relative name stored in .list

    # Decision files reference SEG-Y by relative name so they stay portable
    decision_root = os.path.join(output_dir, out)
    create_decision_files(
        shots,
        out=decision_root,
        word_sx=w_sx, word_sy=w_sy, word_sz=w_sz,
        word_rx=w_rx, word_ry=w_ry, word_rz=w_rz,
        segy_files=segy_files,
    )

    n_traces = sum(len(s["rcv"]) for s in shots)
    print(f"  {len(shots)} shots  ×  {len(shots[0]['rcv'])} receivers"
          f"  =  {n_traces} traces total")
    print(f"  Samples/trace: {n_samples}   dt: {dt_ms:.3f} ms ({dt_us} µs)")
    print(f"  Output dir   : {os.path.abspath(output_dir)}")
