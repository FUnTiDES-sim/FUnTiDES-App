"""
Create decision files from acquisition geometry.

A decision file pair stores per-trace SEG-Y header words for every
source–receiver pair (mapped file) plus an index of unique sources
(index file), both as big-endian binary tables.

Binary format
-------------
Header  (always 60 bytes = 15 × int32 big-endian):
    [0]        : 0  (unused)
    [1]        : n_record
    [2]        : n_word  (words stored per record)
    [3..2+n_word] : word_numbers  (which SEG-Y word numbers are in the record)
    [remaining until word 15] : 1  (padding to always fill 15 header words)

Records (n_word×4 + 4 + 8 bytes each):
    n_word × int32  : word values
    int32           : list_index  (0-based index into the .list file)
    int64           : offset_index (0-based trace index in that file)

.list file
----------
One filename per line.
  mapped.list : SEG-Y files (one per shot or one shared file)
  index.list  : single entry → the mapped decision rootname

Usage
-----
Library::

    from makutu.acquisition.decision_create import create_decision_files
    create_decision_files(
        shots,           # list of {"src": [x,y,z], "rcv": [[x,y,z], ...]}
        out="survey",
        word_sx=22, word_sy=23, word_sz=14,
        word_rx=24, word_ry=25, word_rz=13,
        segy_files=["shot_1.segy", "shot_2.segy"],  # optional
    )

CLI::

    python -m makutu.acquisition.decision_create geometry.traces --out survey
    python -m makutu.acquisition.decision_create geometry.traces --out survey \\
        --segy shot_1.segy shot_2.segy
    python -m makutu.acquisition.decision_create --from-segy survey.segy --out survey
    python -m makutu.acquisition.decision_create --from-segy shot_1.segy shot_2.segy --out survey

Traces file format (geometry.traces)::

    22 23 14 24 25 13          # header: word numbers (space-separated)
    3991 4991 201 201 5991 301  # one line per trace
    ...

Coordinates are rounded to the nearest integer (consistent with how
integer truncation of float coordinates).
"""

import argparse
import os
import struct
import sys

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Low-level binary I/O
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_TOTAL_WORDS = 15  # header is always 60 bytes = 15 × int32


def _write_header(fh, n_record, word_numbers):
    n_word = len(word_numbers)
    padding = _HEADER_TOTAL_WORDS - 3 - n_word  # 12 - n_word  ones
    if padding < 0:
        raise ValueError(
            f"Too many word_numbers ({n_word}); max is 12 for a 60-byte header."
        )
    fh.write(struct.pack(">iii", 0, n_record, n_word))
    for w in word_numbers:
        fh.write(struct.pack(">i", w))
    for _ in range(padding):
        fh.write(struct.pack(">i", 1))


def _write_record(fh, word_values, list_index, offset_index):
    """word_values: iterable of int, in the same order as word_numbers."""
    for v in word_values:
        fh.write(struct.pack(">i", int(v)))
    fh.write(struct.pack(">i", int(list_index)))
    fh.write(struct.pack(">q", int(offset_index)))


# ─────────────────────────────────────────────────────────────────────────────
# File-level writers
# ─────────────────────────────────────────────────────────────────────────────

def write_mapped(rootname, shots, mapped_word_numbers,
                 src_words, rcv_words, segy_files=None):
    """
    Write ``<rootname>.mapped.data`` and ``<rootname>.mapped.list``.

    Parameters
    ----------
    rootname : str
        Output rootname (no extension).
    shots : list of dict
        Each dict has keys ``"src"`` (list/array [x,y,z]) and
        ``"rcv"`` (list of [x,y,z] arrays, one per receiver).
    mapped_word_numbers : list of int
        SEG-Y word numbers to store per trace (in order).
    src_words : tuple of int
        (word_x, word_y, word_z) for the source.
    rcv_words : tuple of int
        (word_x, word_y, word_z) for the receivers.
    segy_files : list of str or None
        One SEG-Y filename per shot.  If *None*, a placeholder entry
        ``"<rootname>_shot_N.segy"`` is generated for each shot.

    Returns
    -------
    list of int
        For each shot: the 0-based trace index in the mapped file where
        that shot's first trace appears (used by write_index).
    """
    wn = mapped_word_numbers
    total_traces = sum(len(s["rcv"]) for s in shots)

    # Build SEG-Y list
    if segy_files is None:
        segy_files = [
            f"{os.path.basename(rootname)}_shot_{i + 1}.segy"
            for i in range(len(shots))
        ]
    if len(segy_files) != len(shots):
        raise ValueError(
            f"segy_files has {len(segy_files)} entries but there are "
            f"{len(shots)} shots."
        )

    # Write .list
    with open(rootname + ".mapped.list", "w") as fh:
        for fname in segy_files:
            fh.write(fname + "\n")

    # Build a lookup: word_number → column index in word_values tuple
    wn_set = set(wn)
    sw_x, sw_y, sw_z = src_words
    rw_x, rw_y, rw_z = rcv_words

    # Track where each shot starts in the mapped file (for index writer)
    shot_offsets = []
    trace_counter = 0

    with open(rootname + ".mapped.data", "wb") as fh:
        _write_header(fh, total_traces, wn)

        for shot_idx, shot in enumerate(shots):
            shot_offsets.append(trace_counter)
            sx, sy, sz = [int(v) for v in shot["src"]]

            for rcv_idx, rcv in enumerate(shot["rcv"]):
                rx, ry, rz = [int(v) for v in rcv]

                # Build word-value dict for this trace
                vals = {}
                if sw_x in wn_set:
                    vals[sw_x] = sx
                if sw_y in wn_set:
                    vals[sw_y] = sy
                if sw_z in wn_set:
                    vals[sw_z] = sz
                if rw_x in wn_set:
                    vals[rw_x] = rx
                if rw_y in wn_set:
                    vals[rw_y] = ry
                if rw_z in wn_set:
                    vals[rw_z] = rz

                for w, v in shot.get("words", {}).items():
                    if w in wn_set and w not in vals:
                        vals[w] = int(v[rcv_idx] if np.ndim(v) else v)

                row = [vals.get(w, 0) for w in wn]
                _write_record(fh, row, shot_idx, rcv_idx)
                trace_counter += 1

    return shot_offsets


def write_index(rootname, shots, index_word_numbers,
                src_words, shot_offsets, mapped_rootname):
    """
    Write ``<rootname>.index.data`` and ``<rootname>.index.list``.

    Parameters
    ----------
    rootname : str
        Output rootname (no extension).
    shots : list of dict
        Same list passed to :func:`write_mapped`.
    index_word_numbers : list of int
        SEG-Y word numbers to store per source record (typically source
        X/Y/Z words only).
    src_words : tuple of int
        (word_x, word_y, word_z) for the source.
    shot_offsets : list of int
        0-based trace offset in the mapped file for each shot
        (returned by :func:`write_mapped`).
    mapped_rootname : str
        Rootname of the mapped decision file to reference in
        ``<rootname>.index.list``.
    """
    user_wn = index_word_numbers    # words actually stored per record
    sw_x, sw_y, sw_z = src_words
    wn_set = set(user_wn)
    n_shots = len(shots)

    # The header declares n_word = 2 (system) + len(user_wn).
    # Words 3 and 4 are internal linkage words prepended to the header
    # word list but not stored in the per-record data.
    header_wn = [3, 4] + list(user_wn)

    # .list points to the mapped decision file
    with open(rootname + ".index.list", "w") as fh:
        fh.write(os.path.basename(mapped_rootname) + ".mapped\n")

    with open(rootname + ".index.data", "wb") as fh:
        _write_header(fh, n_shots, header_wn)

        for shot_idx, shot in enumerate(shots):
            sx, sy, sz = [int(v) for v in shot["src"]]
            vals = {}
            if sw_x in wn_set:
                vals[sw_x] = sx
            if sw_y in wn_set:
                vals[sw_y] = sy
            if sw_z in wn_set:
                vals[sw_z] = sz

            # Only store user_wn values in the record (not the system words)
            row = [vals.get(w, 0) for w in user_wn]
            # list_index = 0 (index.list has only one entry: the mapped file)
            # offset_index = position of this shot's first trace in mapped
            _write_record(fh, row, 0, shot_offsets[shot_idx])


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_decision_files(shots, out,
                          word_sx=22, word_sy=23, word_sz=14,
                          word_rx=24, word_ry=25, word_rz=13,
                          extra_mapped_words=None,
                          index_words=None,
                          segy_files=None):
    """
    Create a complete set of decision files from acquisition geometry.

    Parameters
    ----------
    shots : list of dict
        Each dict must contain:
          - ``"src"``: array-like [x, y, z] for the source position
          - ``"rcv"``: list of array-like [x, y, z] for receiver positions
    out : str
        Output rootname.  Files created:
        ``<out>.mapped.data``, ``<out>.mapped.list``,
        ``<out>.index.data``, ``<out>.index.list``.
    word_sx, word_sy, word_sz : int
        SEG-Y word numbers for source X, Y, Z.
    word_rx, word_ry, word_rz : int
        SEG-Y word numbers for receiver X, Y, Z.
    extra_mapped_words : list of int, optional
        Additional word numbers to include in the mapped file (beyond
        the six src/rcv words).  Values will be stored as 0 unless the
        shot dict contains a ``"words"`` key mapping word → value (one
        value for the whole shot, or a sequence with one value per receiver).
    index_words : list of int, optional
        Word numbers to store in the index file.  Defaults to
        ``[word_sx, word_sy, word_sz]``.
    segy_files : list of str or None
        One SEG-Y filename per shot.
    """
    src_words = (word_sx, word_sy, word_sz)
    rcv_words = (word_rx, word_ry, word_rz)

    # Mapped file stores all six coordinate words (+ any extras)
    base_mapped = [word_sx, word_sy, word_sz, word_rx, word_ry, word_rz]
    if extra_mapped_words:
        for w in extra_mapped_words:
            if w not in base_mapped:
                base_mapped.append(w)

    if index_words is None:
        index_words = [word_sx, word_sy, word_sz]

    shot_offsets = write_mapped(
        out, shots, base_mapped, src_words, rcv_words, segy_files
    )
    write_index(out, shots, index_words, src_words, shot_offsets, out)

    n_traces = sum(len(s["rcv"]) for s in shots)
    print(
        f"Created {out}.mapped.data  ({len(shots)} shots, {n_traces} traces, "
        f"words={base_mapped})"
    )
    print(
        f"Created {out}.index.data   ({len(shots)} sources, "
        f"words={index_words})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y geometry reader  (pure-Python, no segyio required)
# ─────────────────────────────────────────────────────────────────────────────

def read_segy_headers(path, word_numbers):
    """
    Read selected trace-header words from a SEG-Y Rev-0/Rev-1 file.

    Word N is the int32 at byte offset
    ``(N-1)*4`` (0-indexed) within the 240-byte trace header.

    Parameters
    ----------
    path : str
        Path to the SEG-Y file.
    word_numbers : list of int
        Which header words to extract.

    Returns
    -------
    list of dict  {word_number: int_value, ...}  – one dict per trace
    """
    with open(path, "rb") as fh:
        data = fh.read()

    bh         = data[3200:3600]
    n_samples  = struct.unpack(">H", bh[20:22])[0]
    trace_size = 240 + n_samples * 4
    n_traces   = (len(data) - 3600) // trace_size

    rows = []
    for i in range(n_traces):
        th  = data[3600 + i * trace_size : 3600 + i * trace_size + 240]
        row = {w: struct.unpack(">i", th[(w - 1) * 4:(w - 1) * 4 + 4])[0]
               for w in word_numbers}
        rows.append(row)
    return rows


def shots_from_segy(path,
                    segy_src_words=(19, 20, 13),
                    segy_rcv_words=(21, 22, 11),
                    out_src_words=(22, 23, 14),
                    out_rcv_words=(24, 25, 13)):
    """
    Extract acquisition geometry from a SEG-Y file and return shots.

    Reads source/receiver coordinates from the SEG-Y trace headers and
    remaps them to the output decision-file word numbers.  Traces sharing
    the same source position are grouped into a single shot.

    Parameters
    ----------
    path : str
        SEG-Y file path.
    segy_src_words : tuple of int
        Header words to read as source (X, Y, Z) from the SEG-Y.
        Default: 19=sx, 20=sy, 13=sdepth (source depth, positive down).
    segy_rcv_words : tuple of int
        Header words to read as receiver (X, Y, Z) from the SEG-Y.
        Default: 21=gx, 22=gy, 11=gelev, which holds the receiver depth
        (positive down) in the files written by makutu.
    out_src_words : tuple of int
        Decision-file word numbers to assign source (X, Y, Z).
    out_rcv_words : tuple of int
        Decision-file word numbers to assign receiver (X, Y, Z).

    Returns
    -------
    shots : list of dict  {"src": [x,y,z], "rcv": [[x,y,z], ...]}
    segy_files : list of str  (just [path] – all traces in one file)
    """
    all_words = list(dict.fromkeys(list(segy_src_words) + list(segy_rcv_words)))
    raw_rows  = read_segy_headers(path, all_words)

    # Remap SEG-Y words → output decision words
    remapped = []
    for row in raw_rows:
        r = {}
        for segy_w, out_w in zip(segy_src_words, out_src_words):
            r[out_w] = row.get(segy_w, 0)
        for segy_w, out_w in zip(segy_rcv_words, out_rcv_words):
            r[out_w] = row.get(segy_w, 0)
        remapped.append(r)

    shots = shots_from_traces(remapped, list(out_src_words) + list(out_rcv_words),
                              src_words=out_src_words,
                              rcv_words=out_rcv_words)
    return shots, [path]


def shots_from_segy_list(paths,
                         segy_src_words=(19, 20, 13),
                         segy_rcv_words=(21, 22, 11)):
    """
    Extract acquisition geometry from multiple SEG-Y files, one shot per file.

    Each file is treated as a single shot: the source position is read from
    the first trace header, and all traces contribute receiver positions.

    Parameters
    ----------
    paths : list of str
        SEG-Y file paths, one per shot.
    segy_src_words : tuple of int
        Header words to read as source (X, Y, Z) from each SEG-Y file.
    segy_rcv_words : tuple of int
        Header words to read as receiver (X, Y, Z) from each SEG-Y file.

    Returns
    -------
    shots : list of dict  {"src": [x,y,z], "rcv": [[x,y,z], ...]}
    segy_files : list of str  (same as *paths*, one entry per shot)
    """
    all_words = list(dict.fromkeys(list(segy_src_words) + list(segy_rcv_words)))
    shots = []
    for path in paths:
        raw_rows = read_segy_headers(path, all_words)
        if not raw_rows:
            raise ValueError(f"No traces found in {path}")

        # Source from first trace
        first = raw_rows[0]
        src = [first.get(w, 0) for w in segy_src_words]

        # Receivers from all traces
        rcv = [[row.get(w, 0) for w in segy_rcv_words] for row in raw_rows]

        # Remap to output word numbers (values stay the same, just labelled)
        shots.append({"src": src, "rcv": rcv})

    return shots, list(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Traces-file reader
# ─────────────────────────────────────────────────────────────────────────────

def read_traces_file(path):
    """
    Parse a traces file as produced by ``decision_create_light_mpi``.

    Format::

        22 23 14 24 25 13         # first line: column word-numbers
        3991 4991 201 201 5991 301  # subsequent lines: one trace per row

    Returns
    -------
    word_numbers : list of int
    rows : list of dict  {word_number: int_value, ...}
    """
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    word_numbers = list(map(int, lines[0].split()))
    rows = []
    for line in lines[1:]:
        vals = list(map(float, line.split()))
        rows.append({w: int(v) for w, v in zip(word_numbers, vals)})
    return word_numbers, rows


def shots_from_traces(rows, word_numbers,
                      src_words=(22, 23, 14), rcv_words=(24, 25, 13)):
    """
    Group a flat list of trace dicts (from :func:`read_traces_file`) into
    a list of shot dicts ``{"src": [x,y,z], "rcv": [[x,y,z], ...]}``.

    Traces are grouped by source position (src_words values).
    """
    sw_x, sw_y, sw_z = src_words
    rw_x, rw_y, rw_z = rcv_words

    shots_dict = {}  # (sx,sy,sz) → {"src":..., "rcv":[...]}
    for row in rows:
        key = (row.get(sw_x, 0), row.get(sw_y, 0), row.get(sw_z, 0))
        if key not in shots_dict:
            shots_dict[key] = {
                "src": list(key),
                "rcv": [],
            }
        shots_dict[key]["rcv"].append([
            row.get(rw_x, 0),
            row.get(rw_y, 0),
            row.get(rw_z, 0),
        ])

    return list(shots_dict.values())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "traces", nargs="?",
        help="Traces file: first line = word numbers, subsequent lines = "
             "one trace per row.",
    )
    src.add_argument(
        "--from-segy", nargs="+", metavar="FILE",
        help="Read geometry from SEG-Y trace headers.  Pass one file to "
             "group traces by source position into shots; pass multiple "
             "files (one per shot) for one-shot-per-file mode.",
    )

    p.add_argument(
        "--out", required=True,
        help="Output rootname (e.g. 'survey' → survey.mapped.data, etc.).",
    )
    p.add_argument(
        "--segy", nargs="+", metavar="FILE",
        help="SEG-Y files, one per shot (in source-order).  Used as the "
             "file list in the mapped .list file.  When --from-segy is "
             "given, defaults to that same file.",
    )

    # Decision-file word numbers for output
    p.add_argument("--word-sx", type=int, default=22)
    p.add_argument("--word-sy", type=int, default=23)
    p.add_argument("--word-sz", type=int, default=14)
    p.add_argument("--word-rx", type=int, default=24)
    p.add_argument("--word-ry", type=int, default=25)
    p.add_argument("--word-rz", type=int, default=13)
    p.add_argument(
        "--index-words", type=lambda s: list(map(int, s.split(","))),
        default=None,
        metavar="W1,W2,...",
        help="Comma-separated word numbers for the index file.  "
             "Default: source X/Y/Z words.",
    )

    # SEG-Y header word positions (only used with --from-segy)
    p.add_argument("--segy-word-sx", type=int, default=19,
                   help="SEG-Y header word for source X  (default: 19).")
    p.add_argument("--segy-word-sy", type=int, default=20,
                   help="SEG-Y header word for source Y  (default: 20).")
    p.add_argument("--segy-word-sz", type=int, default=13,
                   help="SEG-Y header word for the source depth (default: 13, sdepth).")
    p.add_argument("--segy-word-rx", type=int, default=21,
                   help="SEG-Y header word for receiver X (default: 21).")
    p.add_argument("--segy-word-ry", type=int, default=22,
                   help="SEG-Y header word for receiver Y (default: 22).")
    p.add_argument("--segy-word-rz", type=int, default=11,
                   help="SEG-Y header word for the receiver depth (default: 11, gelev as written by makutu).")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    src_words = (args.word_sx, args.word_sy, args.word_sz)
    rcv_words = (args.word_rx, args.word_ry, args.word_rz)

    if args.from_segy:
        segy_src = (args.segy_word_sx, args.segy_word_sy, args.segy_word_sz)
        segy_rcv = (args.segy_word_rx, args.segy_word_ry, args.segy_word_rz)
        if len(args.from_segy) == 1:
            shots, auto_segy = shots_from_segy(
                args.from_segy[0],
                segy_src_words=segy_src,
                segy_rcv_words=segy_rcv,
                out_src_words=src_words,
                out_rcv_words=rcv_words,
            )
        else:
            shots, auto_segy = shots_from_segy_list(
                args.from_segy,
                segy_src_words=segy_src,
                segy_rcv_words=segy_rcv,
            )
        segy_files = args.segy or auto_segy
    else:
        word_numbers, rows = read_traces_file(args.traces)
        shots = shots_from_traces(rows, word_numbers, src_words, rcv_words)
        segy_files = args.segy

    create_decision_files(
        shots,
        out=args.out,
        word_sx=args.word_sx,
        word_sy=args.word_sy,
        word_sz=args.word_sz,
        word_rx=args.word_rx,
        word_ry=args.word_ry,
        word_rz=args.word_rz,
        index_words=args.index_words,
        segy_files=segy_files,
    )


if __name__ == "__main__":
    main()
