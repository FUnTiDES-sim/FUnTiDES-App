"""Acquisitions classes

Coordinates are ``[x, y, depth]`` in metres: the depth is positive downwards from
the free surface (the top face of the mesh). The solver places a point at the mesh
height ``z = lz - depth`` (``WaveSolver.updateSourceAndReceiversFromCoords``).
In SEG-Y files the source depth is ``sdepth`` (byte 49) and makutu writes the
receiver depth in ``gelev`` (byte 41).
"""

from .Acquisition import Acquisition
from .EquispacedAcquisition import EQUISPACEDAcquisition
from .SegyAcquisition import SEGYAcquisition
from .DecisionAcquisition import DecisionAcquisition

from .decision_reader import read_decision_file
from .decision_create import create_decision_files

from .Shot import (
    Shot,
    SourceSet,
    ReceiverSet,
    Source,
    Receiver,
    ShotPoint,
    ShotPointSet
)
