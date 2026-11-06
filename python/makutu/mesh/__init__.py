"""Mesh"""

from .InternalMesh import StructuredMesh, UnstructuredMesh, InternalMesh
from .CartesianModel2D import CartesianMesh2D, build_cartesian_2d_model, sample_sep_2d, gll_points
from .sizing import suggest_mesh, gll_min_spacing

try:
    from .VtkMesh import VTKMesh, VTKSubMesh
    from .VtkFieldSpecifications import VTKPointSpecifications, VTKCellSpecifications, VTKFieldSpecifications
except ImportError:
    pass