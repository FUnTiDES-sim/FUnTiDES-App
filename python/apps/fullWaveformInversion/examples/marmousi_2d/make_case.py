"""
Synthetic FWI on the 2-D Marmousi model.

Writes, in the output directory:
- true.H : Marmousi Vp (FUnTiDES-data/SEP/marmousi/Vp_xz.H, 10 km x 3.48 km, 20 m);
- init.H : initial model, the true model smoothed by a Gaussian filter (``--smooth`` m)
  below the water bottom, the water (1500 m/s, 440 m) kept;
- fwi_parameters.txt : parameters of fwi.py (element size and time step fixed, so that the
  observed data and the inversion use the same mesh and time step; water not updated);
- obs/seismoTrace_shot<id>.sgy : observed data computed on the true model.

Usage
-----
    python make_case.py [--outdir marmousi_2d_run] [--nshots 16] [--freqs 4,6,9] [--maxiter 10]
    cd marmousi_2d_run && python <apps>/fullWaveformInversion/fwi.py --param_file fwi_parameters.txt
    python <this directory>/check_result.py --rundir marmousi_2d_run --png result.png
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))       # fwi.py (initialises Kokkos)
import fwi                                               # noqa: E402

import kokkos                                            # noqa: E402

from makutu.acquisition import Acquisition               # noqa: E402
from makutu.forwardBackward import directSimulation      # noqa: E402
from makutu.imaging import globals as gb                 # noqa: E402
from makutu.input.createSepModel import _write_sep       # noqa: E402
from makutu.input.SEPReader import read_sep              # noqa: E402
from makutu.mesh import suggest_mesh                     # noqa: E402

DATA = os.environ.get("FUNTIDES_DATA", os.path.join(HERE, *[".."] * 6, "FUnTiDES-data"))
VWATER = 1500.0

PARAMETERS = """# Synthetic Marmousi 2-D case written by make_case.py
vpFile=init.H
order={order}
elemSize={elem}
dt={dt}
maxtime={maxtime}
dtSeismo={dtseismo}
sourceType=ricker2
sourceFreq={f0}
dataDirectory=obs
model=c
method={method}
gDn=0
freqFilterList={freqs}
minDepth={water}
gradEdge={edge}
gradTaper={edge}
illumination=true
minimizerOptions={{"maxiter": {maxiter}, "minModel": 1400, "maxModel": 5000}}
"""


def main():
    p = argparse.ArgumentParser("Synthetic Marmousi 2-D FWI case")
    p.add_argument("--outdir", default="marmousi_2d_run")
    p.add_argument("--vp", default=os.path.join(DATA, "SEP", "marmousi", "Vp_xz.H"))
    p.add_argument("--smooth", type=float, default=300.0, help="Gaussian smoothing of the initial model [m]")
    p.add_argument("--f0", type=float, default=8.0, help="Ricker peak frequency [Hz]")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--elem", type=float, default=50.0, help="Element size [m]")
    p.add_argument("--maxtime", type=float, default=3.0)
    p.add_argument("--nshots", type=int, default=16)
    p.add_argument("--rcv-dx", type=float, default=20.0)
    p.add_argument("--freqs", default="4,6,9", help="freqFilterList of fwi.py")
    p.add_argument("--method", default="CGPR")
    p.add_argument("--maxiter", type=int, default=10, help="Iterations per frequency")
    p.add_argument("--edge", type=float, default=100.0, help="gradEdge and gradTaper [m]")
    a = p.parse_args()

    sep = read_sep(a.vp)
    vtrue = sep.data[0].astype(np.float64)                    # (nz, nx), z down
    d = float(sep.d1)
    lx, lz = sep.n1 * d, sep.n2 * d
    water = vtrue <= VWATER + 0.5
    zwater = int(water.sum(axis=0).min())                     # flat water bottom (rows)
    v0 = gaussian_filter(vtrue, a.smooth / d, mode="nearest")
    v0[water] = VWATER

    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)
    _write_sep("true.H", vtrue[None], d, d, 1.0)
    _write_sep("init.H", v0[None], d, d, 1.0)

    # Time step of the true model (highest velocity), shared by the data and the inversion
    _, dt, _ = suggest_mesh((lx, lz), vtrue.min(), vtrue.max(), a.f0, a.order, elem_size=a.elem)
    dt = float(f"{dt:.3g}")
    with open("fwi_parameters.txt", "w") as f:
        f.write(PARAMETERS.format(order=a.order, elem=a.elem, dt=dt, maxtime=a.maxtime, dtseismo=4 * dt,
                                  f0=a.f0, method=a.method, freqs=a.freqs, water=zwater * d, edge=a.edge,
                                  maxiter=a.maxiter))

    wf = fwi.parse_workflow_parameters("fwi_parameters.txt")
    solver, m = fwi.build_solver(wf, "true.H")
    sources = np.linspace(0.03 * lx, 0.97 * lx, a.nshots)
    receivers = np.arange(a.rcv_dx, lx - a.rcv_dx / 2, a.rcv_dx)
    acq = Acquisition(sources=[[x, 0.0, 10.0] for x in sources],
                      receivers=[[x, 0.0, 20.0] for x in receivers])
    gb.initializeGlobalVariables(solver, acq, [acq], 0.0, 1e9, 0.0, ["all"])
    directSimulation(0, exportSeismicTrace=True, traceDirectory="obs")
    for shot in acq.shots:
        os.replace(f"obs/seismoTrace_shot{shot.id}_{int(a.f0)}.sgy", f"obs/seismoTrace_shot{shot.id}.sgy")
    print(f"Observed data: {len(acq.shots)} shots, {len(receivers)} receivers in {os.path.abspath('obs')}")
    gb.solver = None
    del solver, m


if __name__ == "__main__":
    main()
    kokkos.finalize()
