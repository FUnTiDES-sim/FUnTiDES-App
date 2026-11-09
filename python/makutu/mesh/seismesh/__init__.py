"""seismesh : maillages SEM adaptés à un modèle de vitesses (2-D quadrangles, 3-D hexaèdres).

:func:`build_mesh_from_sep` : modèle SEP + fréquence de la source -> ``mesh.msh`` et
``materials.csv`` pour :class:`makutu.mesh.GmshMesh`. Voir README.md.
"""

from .build import build_mesh_from_sep
from .sep import grid_from_sep
from .velocity import VelocityGrid, synthetic_model
