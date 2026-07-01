"""Mesh"""

from .InternalMesh import StructuredMesh, UnstructuredMesh, InternalMesh

try:
    from .VtkMesh import VTKMesh, VTKSubMesh
    from .VtkFieldSpecifications import VTKPointSpecifications, VTKCellSpecifications, VTKFieldSpecifications
except ImportError:
    pass