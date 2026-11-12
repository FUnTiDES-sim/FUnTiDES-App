"""Blocs réguliers non conformes recollés par pénalité intérieure (SIPG) : prototype 2-D.

Ne dépend que de NumPy et SciPy (pas de pyfuntides) : importable sans GPU.
"""
from .solver2d import Mortar2D, Patch, patch_grid, gll, lagrange, derivative, ricker
from .blocks2d import BlockPlan, grid_sampler, plan_blocks, plan_blocks_global, uniform_plan


def __getattr__(name):
    # BlocksMesh2D needs pyfuntides (GPU): imported on demand only
    if name == "BlocksMesh2D":
        from .mesh2d import BlocksMesh2D
        return BlocksMesh2D
    raise AttributeError(name)
