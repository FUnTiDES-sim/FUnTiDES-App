"""Solvers classes"""

from .Solver import Solver
from .AcousticSolver import AcousticSolver
from .ElasticSolver import ElasticSolver
from .WaveSolver import WaveSolver
from .BoundaryComm import BoundaryComm

from .utils.solverutils import printSolver

from .ModelManagement import (
    geos_params_to_model,
    geos_params_from_model,
    grad_to_model,
    grad_from_model,
    min_max_to_model,
    min_max_from_model
)
