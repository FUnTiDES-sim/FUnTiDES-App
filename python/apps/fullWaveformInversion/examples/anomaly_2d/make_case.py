"""
Synthetic 2-D FWI case: two velocity anomalies above a known reflector.

Writes, in the output directory:
- true.H, init.H : true model (2000 m/s, +250 m/s and -250 m/s Gaussian anomalies,
  2600 m/s below 800 m) and initial model (the same without the anomalies), SEP, 2-D
  (n1 = x, n2 = z). The reflector sends back through the anomalies the waves that
  carry their traveltime shifts (smooth anomalies alone scatter little energy back);
- fwi_parameters.txt : parameters of fwi.py, with the element size and the time step
  fixed so that the observed data and the inversion use the same mesh and time step;
- obs/seismoTrace_shot<id>.sgy : observed data computed on the true model
  (8 shots, 73 receivers, positions in the SEG-Y headers).

Usage
-----
    python make_case.py [--outdir anomaly_2d_run] [--method steepestDescent] [--maxiter 10]
    cd anomaly_2d_run && python <apps>/fullWaveformInversion/fwi.py --param_file fwi_parameters.txt
    python <this directory>/check_result.py --rundir anomaly_2d_run
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

from case import (LX, LZ, ANOMALIES, F0, ORDER, ELEM, SOURCES_X, RECEIVERS_X,  # noqa: E402
                  DEPTH_SRC, DEPTH_RCV, MAXTIME, models)

D = 12.5                                  # SEP sampling [m]


PARAMETERS = """# Synthetic 2-D case written by make_case.py
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
minimizerOptions={{"maxiter": {maxiter}, "minls": 2, "minModel": 1500, "maxModel": 3500}}
{extra}"""


def main():
    p = argparse.ArgumentParser("Synthetic 2-D FWI case")
    p.add_argument("--outdir", default="anomaly_2d_run")
    p.add_argument("--method", default="steepestDescent")
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--extra", default="", help="Extra lines of the parameter file (';' separated)")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    os.chdir(a.outdir)
    _, _, v0, vtrue = models(D)
    _write_sep("init.H", v0[None], D, D, 1.0)
    _write_sep("true.H", vtrue[None], D, D, 1.0)

    # Time step of the true model (highest velocity), shared by data and inversion
    _, dt, _ = suggest_mesh((LX, LZ), vtrue.min(), vtrue.max(), F0, ORDER, elem_size=ELEM)
    dt = float(f"{dt:.3g}")
    with open("fwi_parameters.txt", "w") as f:
        f.write(PARAMETERS.format(order=ORDER, elem=ELEM, dt=dt, maxtime=MAXTIME, dtseismo=2 * dt, f0=F0,
                                  method=a.method, maxiter=a.maxiter, extra="\n".join(a.extra.split(";"))))

    # Observed data: same parameters as the inversion, on the true model
    wf = fwi.parse_workflow_parameters("fwi_parameters.txt")
    solver, m = fwi.build_solver(wf, "true.H")
    acq = Acquisition(sources=[[x, 0.0, DEPTH_SRC] for x in SOURCES_X],
                      receivers=[[x, 0.0, DEPTH_RCV] for x in RECEIVERS_X])
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
