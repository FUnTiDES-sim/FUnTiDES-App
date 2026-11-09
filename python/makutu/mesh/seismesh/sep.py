"""Modèles SEP de makutu -> grille de vitesses seismesh (indices (ix, iy, iz), z profondeur)."""
from __future__ import annotations

import os

import numpy as np

from .velocity import VelocityGrid


def grid_from_sep(sep, name="vp", cell_centred=True) -> VelocityGrid:
    """
    Grille seismesh d'un modèle SEP lu par :func:`makutu.input.SEPReader.read_sep`.

    3-D (n3 > 1) : axes 1, 2, 3 = x, y, z. 2-D (n3 == 1) : axe 1 = x, axe 2 = profondeur ;
    la grille a alors ny = 1 (coupe (x, z), y ignoré).

    ``cell_centred`` (défaut, convention de makutu) : l'échantillon i représente la cellule
    [o + i d, o + (i + 1) d) ; le domaine est [o, o + n d] et les interfaces tombent sur les bords
    des cellules. Sinon, les échantillons sont des valeurs ponctuelles et le domaine
    [o, o + (n - 1) d].
    """
    if isinstance(sep, (str, os.PathLike)):
        from makutu.input.SEPReader import read_sep
        sep = read_sep(str(sep))
    data = np.asarray(sep.data, dtype=np.float32)           # (n3, n2, n1)
    c = 0.5 if cell_centred else 0.0                       # samples at the cell centres
    if sep.n3 > 1:
        vel = np.ascontiguousarray(data.transpose(2, 1, 0))  # (n1, n2, n3) = (x, y, z)
        return VelocityGrid(vel, (sep.o1 + c * sep.d1, sep.o2 + c * sep.d2, sep.o3 + c * sep.d3),
                            (sep.d1, sep.d2, sep.d3), name=name, cell_centred=cell_centred)
    vel = np.ascontiguousarray(data[0].T[:, None, :])      # (n1, 1, n2) = (x, -, z)
    return VelocityGrid(vel, (sep.o1 + c * sep.d1, 0.0, sep.o2 + c * sep.d2), (sep.d1, 1.0, sep.d2), name=name,
                        cell_centred=cell_centred)


def is_2d(grid: VelocityGrid) -> bool:
    return grid.shape[1] == 1
