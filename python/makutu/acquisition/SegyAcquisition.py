import os
import sys
import glob
from copy import deepcopy

import numpy as np

from .Acquisition import Acquisition
from .Shot import Source, SourceSet, Receiver, ReceiverSet, Shot
from .decision_reader import Segy, SEGY_READ, p_TRC_HDR


class SEGYAcquisition(Acquisition):
    """
    Acquisition defined from the reading of segy files containing the
    positions of the sources and receivers.
    """
    def __init__(self, mesh=None, dt=None, **kwargs):
        """
        Parameters
        -----------
        mesh : StructuredMesh, or None
            Mesh of the problem (optional).
        dt : float
            Timestep
        **kwargs : keyword arguments
            segdir : str
                Folder containing the .sgy traces
        """
        super().__init__(mesh, dt, **kwargs)
        self.type = "segyAcquisition"

    def acquisition(self, segdir, **kwargs):
        """
        Set the shots configurations

        Parameters
        -----------
        segdir : str
            Folder containing the .sgy traces
        kwargs :
            sbit : tuple of int
                Source bit position in header for x, y and z coordinates.
                Default is (73, 77, 49).
            rbit : tuple of int
                Receiver bit position in header for x, y, z coordinates.
                Default is (81, 85, 41).
        """
        # Byte positions → word numbers: word N is the int32 at byte (N-1)*4
        # Supplied as 1-based byte offsets; convert to word numbers.
        def _byte_to_word(byte):
            return (byte - 1) // 4 + 1

        sbit = kwargs.get("sbit", (73, 77, 49))
        rbit = kwargs.get("rbit", (81, 85, 41))

        assert all(isinstance(s, int) for s in sbit)
        assert all(isinstance(r, int) for r in rbit)

        sw = [_byte_to_word(b) for b in sbit]   # source word numbers
        rw = [_byte_to_word(b) for b in rbit]   # receiver word numbers
        notEmpty = False
        i = 0
        ext = ("*.sgy", "*.segy")
        filesToIgnore = kwargs.get("ignore", ())

        while not notEmpty and i <= len(ext):
            segfiles = glob.glob(os.path.join(segdir, ext[i]))
            if segfiles:
                notEmpty = True
            else:
                if i == 1:
                    print(f"No SEG found in {segdir}")
                    sys.exit(1)
            i += 1

        if len(filesToIgnore):
            for f in filesToIgnore:
                full = os.path.join(segdir, f)
                if full in segfiles:
                    segfiles.remove(full)

        ishot = 1
        shots = []
        for segfile in sorted(segfiles):
            segy = Segy()
            segy.read_header(segfile)
            segy.open(SEGY_READ)

            thdr   = np.zeros(p_TRC_HDR, dtype=np.int32,   order="F")
            dummy  = np.zeros(segy.nt,   dtype=np.float32, order="F")

            # Read scalars from first trace header
            segy.read_trace(1, thdr, dummy, o_irec=1)
            word18 = int(thdr[17])   # 0-based index = word 18 - 1
            # word 18 packs scalel (bytes 69-70) and scalco (bytes 71-72)
            # as two int16 values in big-endian order
            scalel = np.int16((word18 >> 16) & 0xFFFF)
            scalco = np.int16(word18 & 0xFFFF)

            def _scale(val, sc):
                s = float(sc)
                return val * (abs(s) ** np.sign(s)) if s != 0 else float(val)

            # Source from first trace header (assumed constant across file)
            sourceX = _scale(int(thdr[sw[0] - 1]), scalco)
            sourceY = _scale(int(thdr[sw[1] - 1]), scalco)
            sourceZ = _scale(int(thdr[sw[2] - 1]), scalel)
            source  = Source(sourceX, sourceY, sourceZ)

            receiverList = []
            for n in range(1, segy.n_trace + 1):
                segy.read_trace(1, thdr, dummy, o_irec=n)
                receiverX = _scale(int(thdr[rw[0] - 1]), scalco)
                receiverY = _scale(int(thdr[rw[1] - 1]), scalco)
                receiverZ = _scale(int(thdr[rw[2] - 1]), scalel)
                receiverList.append(Receiver(receiverX, receiverY, receiverZ))

            segy.close()

            sourceSet  = SourceSet()
            sourceSet.append(source)
            receiverSet = ReceiverSet(receiverList)
            shots.append(Shot(sourceSet, receiverSet, f"{ishot:05d}"))
            ishot += 1

        self.shots = shots
