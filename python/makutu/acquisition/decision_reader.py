"""
Standalone decision-file reader.

Binary format (identical to decision_create.py):

Header (60 bytes = 15 × int32 big-endian):
    [0]          : 0  (unused)
    [1]          : n_record
    [2]          : n_word  (includes system words 3, 4 for index files)
    [3..2+n_word]: word_numbers
    [remaining]  : 1  (padding to 15 header words)

Records per trace:
    n_stored × int32   word values  (n_stored = n_word for mapped files,
                                     n_word-2 for index files that carry
                                     system words 3/4 in the header only)
    int32              list_index
    int64              offset_index

The actual n_stored is inferred from the file size at open time.
"""

import struct

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEC_READ  = 0
SEGY_READ = 0
p_TRC_HDR = 60   # number of 4-byte header words in a SEG-Y trace header


# ─────────────────────────────────────────────────────────────────────────────
# Decision file reader
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_WORDS = 15        # header is always 60 bytes = 15 × int32
_HEADER_BYTES = _HEADER_WORDS * 4


class Decision:
    """
    Pure-Python reader for decision files (.data).

    Supports both *mapped* files (all n_word values stored per record)
    and *index* files (system words 3, 4 declared in header but not
    stored — records hold only the remaining user words).
    """

    def __init__(self):
        self._path       = None
        self._fh         = None
        self.n_record    = 0
        self.n_word      = 0          # as declared in header
        self._word_nums  = []         # word_numbers from header
        self._n_stored   = 0          # words actually written per record
        self._n_skip     = 0          # leading header words not in records
        self._record_size = 0         # bytes per record

    # ------------------------------------------------------------------
    def read_header(self, rootname):
        """Read binary header from ``<rootname>.data``."""
        path = rootname + ".data"
        with open(path, "rb") as fh:
            raw = fh.read()

        file_size = len(raw)

        # Parse header
        _unused, n_record, n_word = struct.unpack_from(">3i", raw, 0)
        word_nums = list(struct.unpack_from(f">{n_word}i", raw, 12))

        # Infer how many word values are actually stored per record.
        # Try n_word first (mapped), then n_word-2 (index with system words).
        data_bytes = file_size - _HEADER_BYTES
        for n_stored in (n_word, n_word - 2, n_word - 1):
            if n_stored < 0:
                continue
            rs = n_stored * 4 + 4 + 8   # words + list_index(int32) + offset(int64)
            if n_record > 0 and data_bytes % rs == 0 and data_bytes // rs == n_record:
                break
        else:
            # Fallback: assume all words are stored
            n_stored = n_word
            rs = n_word * 4 + 4 + 8

        self._path        = path
        self.n_record     = n_record
        self.n_word       = n_word
        self._word_nums   = word_nums
        self._n_stored    = n_stored
        self._n_skip      = n_word - n_stored   # system words not in records
        self._record_size = rs

    # ------------------------------------------------------------------
    def get_word_position(self, word_number):
        """
        Return the 1-based position of *word_number* in ``words_in``.
        Returns 0 if the word is not in the file.
        """
        try:
            return self._word_nums.index(word_number) + 1
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    def open(self, _mode):
        """Open the data file for reading."""
        self._fh = open(self._path, "rb")

    # ------------------------------------------------------------------
    def read_record(self, _unused, words_in, list_index, offset_index,
                    o_irec=1):
        """
        Read record *o_irec* (1-based) into the provided arrays.

        Parameters
        ----------
        _unused : int
            Ignored (unused, kept for API compatibility).
        words_in : np.ndarray, int32, shape (n_word,)
            Receives word values.  Positions for system words (not stored
            in the file) are set to 0.
        list_index : np.ndarray, int32, shape (1,)
            Receives the SEG-Y file list index.
        offset_index : np.ndarray, int64, shape (1,)
            Receives the trace byte offset in the SEG-Y file.
        o_irec : int
            1-based record number.
        """
        offset = _HEADER_BYTES + (o_irec - 1) * self._record_size
        self._fh.seek(offset)
        raw = self._fh.read(self._record_size)

        # Zero out system-word slots (leading n_skip positions)
        words_in[:self._n_skip] = 0

        # Unpack stored word values into positions [n_skip .. n_word-1]
        vals = struct.unpack_from(f">{self._n_stored}i", raw, 0)
        for i, v in enumerate(vals):
            words_in[self._n_skip + i] = v

        # list_index and offset_index follow the word values
        off_li = self._n_stored * 4
        list_index[0]   = struct.unpack_from(">i", raw, off_li)[0]
        offset_index[0] = struct.unpack_from(">q", raw, off_li + 4)[0]

    # ------------------------------------------------------------------
    def close(self):
        """Close the data file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate database  (identity only)
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Stub coordinate database.

    Only the ``"identity"`` transform is supported (coordinates returned
    unchanged).  Only the ``"identity"`` transform is implemented.
    """

    def init(self, file="identity"):
        if file != "identity":
            raise NotImplementedError(
                f"Non-identity database coordinate conversion is not implemented. "
                f"Database: {file}"
            )

    @staticmethod
    def utmxy2xy(x, y):
        return x, y

    @staticmethod
    def xy2utmxy(x, y):
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y trace reader  (pure-Python, no segyio required)
# ─────────────────────────────────────────────────────────────────────────────

class Segy:
    """
    Minimal SEG-Y reader.

    Reads Rev-0 / Rev-1 SEG-Y files without segyio.
    Word N in the trace header is the int32 at byte offset ``(N-1)*4``
    (0-indexed).
    """

    def __init__(self):
        self._path      = None
        self._data      = None
        self.nt         = 0
        self.dt         = 0.0     # sample interval in microseconds
        self.n_trace    = 0
        self._trace_size = 0

    def read_header(self, filename):
        """Read binary and textual file headers."""
        with open(filename, "rb") as fh:
            self._data = fh.read()

        bh = self._data[3200:3600]
        self.dt      = float(struct.unpack(">H", bh[16:18])[0])   # µs
        self.nt      = struct.unpack(">H", bh[20:22])[0]
        self._trace_size = 240 + self.nt * 4
        self.n_trace = (len(self._data) - 3600) // self._trace_size
        self._path   = filename

    def open(self, _mode):
        """No-op: data is already loaded in read_header."""

    def read_trace(self, _unused, thdr_in, trace_in, o_irec=1):
        """
        Read trace *o_irec* (1-based) header words and sample data.

        Parameters
        ----------
        _unused : int
            Ignored.
        thdr_in : np.ndarray, int32, shape (p_TRC_HDR,)
            Receives the first p_TRC_HDR trace-header words.
        trace_in : np.ndarray, float32, shape (nt,)
            Receives the decompressed sample data (IBM float → IEEE float).
        o_irec : int
            1-based trace index.
        """
        base = 3600 + (o_irec - 1) * self._trace_size
        th   = self._data[base : base + 240]

        # Fill trace header words
        n_hdr = min(len(thdr_in), p_TRC_HDR)
        for i in range(n_hdr):
            thdr_in[i] = struct.unpack_from(">i", th, i * 4)[0]

        # Sample data: SEG-Y Rev-0 uses IBM 32-bit float; Rev-1 may use
        # IEEE float (data sample format code in binary header word 7).
        fmt_code = struct.unpack(">H", self._data[3224:3226])[0]
        sd = self._data[base + 240 : base + 240 + self.nt * 4]
        if fmt_code == 1:
            # IBM floating point
            for i in range(self.nt):
                trace_in[i] = _ibm_to_ieee(
                    struct.unpack_from(">I", sd, i * 4)[0]
                )
        else:
            # IEEE float (format codes 4 or 5) or int32 (code 2) — treat
            # codes 2/3 as int32→float, codes 4/5 as IEEE float32
            if fmt_code in (4, 5):
                vals = struct.unpack_from(f">{self.nt}f", sd)
            else:
                vals = struct.unpack_from(f">{self.nt}i", sd)
            trace_in[:] = vals

    def close(self):
        """Release in-memory data."""
        self._data = None


def _ibm_to_ieee(ibm):
    """Convert a single IBM 32-bit float (big-endian uint32) to Python float."""
    if ibm == 0:
        return 0.0
    sign     = -1 if (ibm >> 31) else 1
    exponent = ((ibm >> 24) & 0x7F) - 64
    mantissa = (ibm & 0x00FFFFFF) / 16777216.0   # 2^24
    return sign * mantissa * (16.0 ** exponent)


# ─────────────────────────────────────────────────────────────────────────────
# Public reader function
# ─────────────────────────────────────────────────────────────────────────────

def read_decision_file(decision_file, database_file="identity",
                       need_traces=False, zopp=False, **kwargs):
    """
    Retrieve source and receiver coordinates (and optionally traces) from a
    decision file.

    Parameters
    ----------
    decision_file : str
        Rootname of the decision file set (without ``.mapped.data`` /
        ``.mapped.list`` suffix).  E.g. ``"survey"`` reads
        ``survey.mapped.data`` and ``survey.mapped.list``.
    database_file : str, optional
        Name of the database file for coordinate conversion.
        Default ``"identity"`` (no conversion).
    need_traces : bool, optional
        Whether to also extract seismic trace data from the associated
        SEG-Y files. Default ``False``.
    zopp : bool, optional
        Flip the sign of Z coordinates. Default ``False``.

    Keyword arguments
    -----------------
    scalco : int
        Scalar applied to all horizontal coordinates.
    scalel : int
        Scalar applied to elevations / depths.
    utmxy2xy : bool
        Convert UTM → local XY using the database.
    xy2utmxy : bool
        Convert local XY → UTM using the database.
    swords : list of int
        Word indices for source X, Y, Z. Default: ``[22, 23, 14]``.
    rwords : list of int
        Word indices for receiver X, Y, Z. Default: ``[24, 25, 13]``.
    dwords : list of int
        Word indices for DAS geometry. Default: ``[60, 61, 62]``.
        Pass this key (even as ``None``) to activate DAS support.
    gather : str
        ``'CSG'`` / ``'csg'`` (common source gather, default) or
        ``'CRG'`` / ``'crg'`` (common receiver gather).
    sorted : bool
        Preserve the receiver ordering from the decision file.

    Returns
    -------
    d_src : dict
        ``{"src1": [x, y, z], "src2": [x, y, z], ...}``
    d_rcv : dict
        ``{"rcv_1": [[x, y, z], ...], "rcv_2": [[x, y, z], ...], ...}``
    n : int
        Number of sources (gather keys).
    """

    # ------------------------------------------------------------------ #
    # Open decision file and read header                                   #
    # ------------------------------------------------------------------ #
    dec = Decision()
    dec.read_header(decision_file + ".mapped")

    n_trace = dec.n_record
    n_word  = dec.n_word

    # ------------------------------------------------------------------ #
    # Resolve word positions                                               #
    # ------------------------------------------------------------------ #
    rw = kwargs.get("rwords") or [24, 25, 13]
    sw = kwargs.get("swords") or [22, 23, 14]

    dargs = "dwords" in kwargs
    if dargs:
        dw = kwargs.get("dwords") or [60, 61, 62]
        rw = rw + dw

    # Gather mode
    gather = kwargs.get("gather", "csg").lower()
    CSG = gather != "crg"

    if CSG:
        gkpos_idx = [dec.get_word_position(w) - 1 for w in sw]
        gmpos_idx = [dec.get_word_position(w) - 1 for w in rw]
        gk_string, gm_string = "src", "rcv"
    else:
        gkpos_idx = [dec.get_word_position(w) - 1 for w in rw]
        gmpos_idx = [dec.get_word_position(w) - 1 for w in sw]
        gk_string, gm_string = "rcv", "src"

    # Scaling: read from kwargs or from decision file
    scalc = "scalco" not in kwargs and "scalel" not in kwargs
    if scalc:
        scalco_idx = dec.get_word_position(21) - 1
        scalel_idx = dec.get_word_position(20) - 1

    if -1 in gkpos_idx + gmpos_idx:
        dec.close()
        raise ValueError(
            "Unable to read source/receiver word positions from the decision "
            "file header. Check that words 22 23 14 (source) and 24 25 13 "
            "(receiver) are present, or pass swords=/rwords= explicitly."
        )

    # ------------------------------------------------------------------ #
    # First pass: loop over all records, build gather dictionary           #
    # ------------------------------------------------------------------ #
    words_in     = np.zeros(n_word, dtype=np.int32, order="F")
    list_index   = np.zeros(1,      dtype=np.int32, order="F")
    offset_index = np.zeros(1,      dtype=np.int64, order="F")
    commonx_gather = {}

    dec.open(DEC_READ)
    for i in range(1, n_trace + 1):
        dec.read_record(1, words_in, list_index, offset_index, o_irec=i)

        gather_key = (
            int(words_in[gkpos_idx[0]]),
            int(words_in[gkpos_idx[1]]),
            int(words_in[gkpos_idx[2]]),
        )

        if need_traces:
            if scalc:
                if CSG and dargs:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]], words_in[gmpos_idx[3]],
                             words_in[gmpos_idx[4]], words_in[gmpos_idx[5]],
                             words_in[scalel_idx], words_in[scalco_idx],
                             list_index[0], offset_index[0])
                else:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]],
                             words_in[scalel_idx], words_in[scalco_idx],
                             list_index[0], offset_index[0])
            else:
                if CSG and dargs:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]], words_in[gmpos_idx[3]],
                             words_in[gmpos_idx[4]], words_in[gmpos_idx[5]],
                             list_index[0], offset_index[0])
                else:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]],
                             list_index[0], offset_index[0])
        else:
            if scalc:
                if CSG and dargs:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]], words_in[gmpos_idx[3]],
                             words_in[gmpos_idx[4]], words_in[gmpos_idx[5]],
                             words_in[scalel_idx], words_in[scalco_idx])
                else:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]],
                             words_in[scalel_idx], words_in[scalco_idx])
            else:
                if CSG and dargs:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]], words_in[gmpos_idx[3]],
                             words_in[gmpos_idx[4]], words_in[gmpos_idx[5]])
                else:
                    entry = (i,
                             words_in[gmpos_idx[0]], words_in[gmpos_idx[1]],
                             words_in[gmpos_idx[2]])

        if gather_key not in commonx_gather:
            commonx_gather[gather_key] = set()
        commonx_gather[gather_key].add(entry)

    # ------------------------------------------------------------------ #
    # If traces needed, read SEG-Y file list                               #
    # ------------------------------------------------------------------ #
    if need_traces:
        with open(decision_file + ".mapped.list") as fh:
            ifiles = [line.rstrip() for line in fh]

    # ------------------------------------------------------------------ #
    # Coordinate scaling helpers                                           #
    # ------------------------------------------------------------------ #
    def scal2indx(x):
        return abs(float(x)) ** np.sign(x)

    crd_conv = lambda x: x
    if database_file != "identity":
        db = Database()
        db.init(file=database_file)
        if kwargs.get("utmxy2xy"):
            crd_conv = lambda x: list(db.utmxy2xy(x[0], x[1])) + [x[2]]
        elif kwargs.get("xy2utmxy"):
            crd_conv = lambda x: list(db.xy2utmxy(x[0], x[1])) + [x[2]]
        else:
            crd_conv = lambda x: list(db.utmxy2xy(x[0], x[1])) + [x[2]]

    if not scalc:
        scalco = kwargs["scalco"]
        scalel = kwargs["scalel"]

    # ------------------------------------------------------------------ #
    # Second pass: convert and collect source / receiver coordinates       #
    # ------------------------------------------------------------------ #
    d_gathermember = {}
    d_gatherkey    = {}
    n              = 0
    segy_idx       = -1

    for gather_key, cm_set in commonx_gather.items():
        n += 1
        cm_gather = sorted(cm_set) if kwargs.get("sorted") else list(cm_set)

        for i_gm, tmp in enumerate(cm_gather):
            if i_gm == 0 and scalc:
                if CSG and dargs:
                    scalco = tmp[7]
                    scalel = tmp[6]
                else:
                    scalco = tmp[5]
                    scalel = tmp[4]

            if i_gm == 0:
                sign_z = -1 if zopp else 1
                if CSG and dargs:
                    scalindx = [scal2indx(scalco), scal2indx(scalco),
                                 scal2indx(scalel) * sign_z,
                                 scal2indx(scalco), scal2indx(scalco),
                                 scal2indx(scalco)]
                else:
                    scalindx = [scal2indx(scalco), scal2indx(scalco),
                                 scal2indx(scalel) * sign_z]

                gk_float = [float(v) * s
                             for v, s in zip(gather_key, scalindx)]
                d_gatherkey[f"{gk_string}{n}"] = crd_conv(gk_float)
                d_gathermember[f"{gm_string}_{n}"] = []

            if CSG and dargs:
                gm_float = [float(v) * s
                             for v, s in zip(tmp[1:7], scalindx)]
            else:
                gm_float = [float(v) * s
                             for v, s in zip(tmp[1:4], scalindx)]
            d_gathermember[f"{gm_string}_{n}"].append(crd_conv(gm_float))

            if need_traces:
                file_idx = tmp[-2]
                if segy_idx != file_idx:
                    if segy_idx != -1:
                        segy_file.close()
                segy_idx  = file_idx
                segy_file = Segy()
                segy_file.read_header(ifiles[segy_idx])
                nt        = segy_file.nt
                dt_segy   = segy_file.dt * 1.0e-6
                trace_in  = np.zeros(nt,       dtype=np.float32, order="F")
                thdr_in   = np.zeros(p_TRC_HDR, dtype=np.int32,  order="F")
                segy_file.open(SEGY_READ)
                segy_file.read_trace(1, thdr_in, trace_in, o_irec=tmp[-1] + 1)

    if segy_idx != -1:
        segy_file.close()
    dec.close()

    return d_gatherkey, d_gathermember, n
