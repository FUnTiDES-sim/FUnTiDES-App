"""Acquisitions classes

Coordinates are ``[x, y, z]`` in metres, seismic convention: z = 0 at the surface
(the free surface, top of the mesh), positive downwards. In SEG-Y files the source
z is ``sdepth`` (byte 49) and makutu writes the receiver z in ``gelev`` (byte 41).
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
