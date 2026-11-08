"""
Synthetic 3-D FWI case: a velocity anomaly above a known reflector.

Writes, in the output directory:
- true.H, init.H : true model (2000 m/s, +250 m/s Gaussian anomaly at 250 m depth,
  2600 m/s below 450 m) and initial model (the same without the anomaly), SEP, 3-D
  (n1 = x, n2 = y, n3 = z);
- fwi_parameters.txt : parameters of fwi.py (element size and time step fixed: the observed
  data and the inversion use the same mesh, inverse crime);
- obs/seismoTrace_shot<id>.sgy : observed data computed on the true model (9 shots on a
  3 x 3 grid, 81 receivers on a 9 x 9 grid, at 25 m depth).

Usage
-----
    python make_case.py [--outdir anomaly_3d_run] [--method CGPR] [--maxiter 8]
    cd anomaly_3d_run && python <apps>/fullWaveformInversion/fwi.py --param_file fwi_parameters.txt
    python <this directory>/check_result.py --rundir anomaly_3d_run --png result.png
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))       # fwi.py (initialises Kokkos)
import fwi                                               # noqa: E402

import kokkos                                            # noqa: E402

from makutu.acquisition import Acquisition               # noqa: E402
from makutu.forwardBackward import directSimulation      # noqa: E402
from makutu.imaging import FWIProblem                 # noqa: E402
from makutu.input.createSepModel import _write_sep       # noqa: E402
from makutu.mesh import suggest_mesh                     # noqa: E402

from case import (LX, LY, LZ, F0, ORDER, ELEM, SOURCES_XY, RECEIVERS_XY,  # noqa: E402
                  DEPTH_SRC, DEPTH_RCV, MAXTIME, models)

D = 25.0                                  # SEP sampling [m]


PARAMETERS = """# Synthetic 3-D case written by make_case.py
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
minDepth=50
gradEdge=100
gradTaper=50
illumination=true
minimizerOptions={{"maxiter": {maxiter}, "minModel": 1500, "maxModel": 3500}}
{extra}"""


def main():
    p = argparse.ArgumentParser("Synthetic 3-D FWI case")
    p.add_argument("--outdir", default="anomaly_3d_run")
    p.add_argument("--method", default="CGPR")
    p.add_argument("--maxiter", type=int, default=8)
    p.add_argument("--extra", default="", help="Extra lines of the parameter file (';' separated)")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)
    _, _, _, v0, vtrue = models(D)
    _write_sep("init.H", v0, D, D, D)
    _write_sep("true.H", vtrue, D, D, D)

    # Time step of the true model (highest velocity), shared by data and inversion
    _, dt, _ = suggest_mesh((LX, LY, LZ), vtrue.min(), vtrue.max(), F0, ORDER, elem_size=ELEM)
    dt = float(f"{dt:.3g}")
    with open("fwi_parameters.txt", "w") as f:
        f.write(PARAMETERS.format(order=ORDER, elem=ELEM, dt=dt, maxtime=MAXTIME, dtseismo=2 * dt, f0=F0,
                                  method=a.method, maxiter=a.maxiter, extra="\n".join(a.extra.split(";"))))

    # Observed data: same parameters as the inversion, on the true model
    wf = fwi.parse_workflow_parameters("fwi_parameters.txt")
    solver, m = fwi.build_solver(wf, "true.H")
    acq = Acquisition(sources=[[x, y, DEPTH_SRC] for x, y in SOURCES_XY],
                      receivers=[[x, y, DEPTH_RCV] for x, y in RECEIVERS_XY])
    problem = FWIProblem(solver, acq, minModel=0.0, maxModel=1e9, freqList=["all"])
    directSimulation(problem, 0, exportSeismicTrace=True, traceDirectory="obs")
    for shot in acq.shots:
        os.replace(f"obs/seismoTrace_shot{shot.id}_{int(F0)}.sgy", f"obs/seismoTrace_shot{shot.id}.sgy")
    print(f"Observed data: {len(acq.shots)} shots in {os.path.abspath('obs')}")
    problem.release()
    del solver, m


if __name__ == "__main__":
    main()
    kokkos.finalize()
