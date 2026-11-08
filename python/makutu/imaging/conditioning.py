"""
Conditioning of the FWI gradient: a diagonal preconditioner for makutu.imaging.Minimizer
(option ``precond``), P g = mask * g / (H + eps max H).

- mask: 0 above ``minDepth`` (sources and receivers, where the gradient is singular),
  0 within ``edge`` of the absorbing edges (sides and bottom; their terms are not
  differentiated), with cosine ramps of width ``taper`` towards the inside;
- H: illumination, energy of the forward field summed over the shots
  (``problem.illumination``, set by makutu.imaging.gradf), the source side of the diagonal
  pseudo-Hessian (Shin et al., 2001); it compensates the geometrical spreading.
  ``eps`` (relative to max H) bounds the gain where H is small.

The Minimizer line searches still use the true gradient.
"""

import numpy as np


def _ramp(d, width):
    """0 for d <= 0, cosine ramp to 1 at d = width, 1 beyond."""
    if width <= 0:
        return (d > 0).astype(float)
    t = np.clip(d / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


class GradientConditioner:
    """
    Diagonal preconditioner of the gradient (callable: ``P(g)``).

    Parameters
    ----------
        solver : AcousticSolver
            Solver of the inversion (Cartesian mesh: lx, lz, ex, ez [, ly, ey])
        minDepth : float, optional
            Depth [m] above which the gradient is set to 0 \
            Default is 0
        edge : float, optional
            Width [m] of the band set to 0 along the sides and the bottom \
            Default is 0
        taper : float, optional
            Width [m] of the cosine ramps of the masks \
            Default is 0
        illumination : bool, optional
            Divide by the illumination ``problem.illumination`` \
            Default is False
        eps : float, optional
            Stabilisation of the division, relative to the maximum illumination \
            Default is 0.01
        problem : makutu.imaging.FWIProblem, optional
            Problem whose illumination is used (required with ``illumination``)
    """
    def __init__(self, solver, minDepth=0.0, edge=0.0, taper=0.0, illumination=False, eps=0.01, problem=None):
        self.minDepth, self.edge, self.taper = float(minDepth), float(edge), float(taper)
        self.illumination, self.eps = bool(illumination), float(eps)
        if self.illumination and problem is None:
            raise ValueError("GradientConditioner: illumination=True needs the problem (problem.illumination)")
        self.problem = problem
        self.mask = self.buildMask(solver)

    def buildMask(self, solver):
        """Mask per element, seismic order (x fastest, z from the surface)."""
        c = lambda n, length: (np.arange(n) + 0.5) * length / n
        if solver.dim == 2:
            Z, X = np.meshgrid(c(solver.ez, solver.lz), c(solver.ex, solver.lx), indexing="ij")
            sides = [X, solver.lx - X]
        else:
            Z, Y, X = np.meshgrid(c(solver.ez, solver.lz), c(solver.ey, solver.ly), c(solver.ex, solver.lx),
                                  indexing="ij")
            sides = [X, solver.lx - X, Y, solver.ly - Y]
        mask = _ramp(Z - self.minDepth, self.taper) if self.minDepth > 0 else np.ones(Z.shape)
        if self.edge > 0:
            for d in sides + [solver.lz - Z]:
                mask = mask * _ramp(d - self.edge, self.taper)
        return mask.ravel()

    def __call__(self, gradJ):
        p = self.mask * gradJ
        if self.illumination:
            H = self.problem.illumination
            if H is None:
                raise RuntimeError("GradientConditioner: no illumination (problem.illumination is set by gradf)")
            p = p / (H + self.eps * np.max(H))
        return p
