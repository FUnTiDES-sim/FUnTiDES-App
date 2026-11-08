"""
Acoustic full-waveform inversion with makutu on the FUnTiDES SEM solvers.

Workflow
--------
1. Initial model: SEP Vp file (2-D if n3 = 1, else 3-D), or a restart model (HDF5).
   It sets the mesh and the time step (``makutu.mesh.mesh_from_sep``).
2. Acquisition: from the trace headers of the observed SEG-Y files (``dataDirectory``,
   standard positions sx, sy, sdepth / gx, gy, gelev, as written by makutu's
   SEGYTraceOutput), or from a decision file (``decision``).
3. For every filter frequency of ``freqFilterList``: minimisation of the least-squares
   cost function (``makutu.imaging.Minimizer``, gradient by the adjoint-state method,
   ``makutu.imaging.gradf``), starting from the model of the previous frequency.

The model is a flat array per element in seismic order (z down), in the parametrisation
``model`` ("c", "1/c" or "1/c2").

With MPI the shots are dealt out to the ranks (every rank holds the whole model and runs the
same optimisation on the summed cost and gradient; rank 0 writes the model files and prints).

Usage
-----
    python fwi.py --param_file fwi_parameters.txt [--dbg | --dbg+]
    mpirun -np 4 python fwi.py --param_file fwi_parameters.txt

Restart (e.g. after velocityModel23.hdf5, written during the 3rd filter frequency):

    python fwi.py --param_file fwi_parameters.txt --restart --rfile velocityModel23.hdf5 \
        --restart_freq 3 --restart_iter 24

The optimisation of that frequency starts again from the model (the conjugate-gradient
history is not kept); the next model files are numbered from 24.

See README.md for the parameters.
"""

import argparse
import os
import sys

from makutu.tools.gpu import select_gpu  # noqa: E402

select_gpu()                  # one GPU per MPI rank of the node, before Kokkos

import numpy as np
import h5py
from mpi4py import MPI

import kokkos
kokkos.initialize()

import pyfuntides.solver as FTSolver

from makutu.acquisition import SEGYAcquisition, DecisionAcquisition
from makutu.forwardBackward import DataSelection
from makutu.forwardBackward.modelio import writeFields
from makutu.imaging import FWIProblem, Minimizer, GradientConditioner
from makutu.mesh import mesh_from_sep
from makutu.solvers import AcousticSolver


def parse_args():
    parser = argparse.ArgumentParser("FWI workflow parser")
    parser.add_argument("--param_file", type=str, dest="pfile", required=True,
                        help="File containing all the parameters necessary")
    parser.add_argument("--r", "--restart", action="store_true", default=False, dest="restart",
                        help="Flag to restart from existing model file")
    parser.add_argument("--rfile", "--restart_model_file", type=str, dest="rfile", default=None,
                        help="HDF5 file (dataset 'velocity') with the model for restart, in the "
                             "parametrisation of this run (c, 1/c or 1/c2)")
    parser.add_argument("--restart_freq", type=int, dest="rfreq", default=1,
                        help="Restart: number (from 1) of the filter frequency of freqFilterList to "
                             "start with (default 1)")
    parser.add_argument("--restart_iter", type=int, dest="riter", default=1,
                        help="Restart: number of the next model file velocityModel<n>.hdf5 (default 1)")
    parser.add_argument("--dbg", dest="DBG", action="store_true", default=False,
                        help="Debug mode: keep the full gradients, the cost functions and the "
                             "partial cost function files")
    parser.add_argument("--dbg+", dest="dbgIO", action="store_true", default=False,
                        help="Debug mode, and export all the seismic traces and residuals")
    args, _ = parser.parse_known_args()
    return args


def str2bool(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def parse_workflow_parameters(pfile):
    """Parameters file: one ``key=value`` per line, ``#`` starts a comment."""
    with open(pfile, "r") as f:
        hdrStr = f.read()

    hdrList = []
    for fl in hdrStr.split('\n'):
        line = fl.split("#")[0].strip()
        if line:
            hdrList += ("--" + line).split("=", 1)

    p = argparse.ArgumentParser("FWI workflow parameters")
    # Model and mesh
    p.add_argument("--vpFile", dest="vpFile", default=None, help="SEP file of the initial Vp model")
    p.add_argument("--rhoFile", dest="rhoFile", default=None, help="SEP file of the density (default 1)")
    p.add_argument("--order", dest="order", type=int, default=2, help="Polynomial order")
    p.add_argument("--ppw", dest="ppw", type=float, default=6.0, help="GLL points per shortest wavelength")
    p.add_argument("--elemSize", dest="elemSize", type=float, default=None, help="Element size [m]")
    p.add_argument("--freeSurface", dest="freeSurface", type=str2bool, default=True,
                   help="Free surface on top")
    p.add_argument("--sponge", dest="sponge", type=float, default=0.0,
                   help="Sponge width [m] on the absorbing edges, on top of the absorbing condition (default 0; useful from about two wavelengths)")
    # Time
    p.add_argument("--maxtime", dest="maxtime", type=float, required=True, help="Recorded time [s]")
    p.add_argument("--dt", dest="dt", type=float, default=None, help="Time step (default from the CFL)")
    p.add_argument("--cfl", dest="cfl", type=float, default=0.3, help="CFL number when dt is not given")
    p.add_argument("--dtSeismo", dest="dtSeismo", type=float, default=None, help="Seismogram sampling [s]")
    p.add_argument("--dtWaveField", dest="dtWaveField", type=float, default=None,
                   help="Interval between the snapshots of the forward run for the gradient [s] "
                        "(default: from the source frequency)")
    p.add_argument("--snapshotCompression", dest="snapshotCompression", choices=["none", "int16"],
                   default="none", help="Storage of the snapshots: float32 (none) or 16-bit integers "
                                        "with a scale per snapshot (int16: half the memory)")
    # Source
    p.add_argument("--sourceType", dest="sourceType", type=str, default="ricker2", help="Source type")
    p.add_argument("--sourceFreq", dest="sourceFreq", type=float, required=True, help="Source frequency [Hz]")
    p.add_argument("--freqFilterList", dest="freqFilterList", type=str, default="all",
                   help="Filter frequencies fq1,fq2,... (default 'all': no filter)")
    # Data and acquisition
    p.add_argument("--dataDirectory", dest="dataDir", type=str, default="seismicTrace",
                   help="Directory of the observed data, one SEG-Y file per shot")
    p.add_argument("--dinfo", dest="dinfo", type=str, default="seismoTrace_shot;5;.sgy",
                   help="Rootname of the data files, digits of the shot id, extension, separated by ';'")
    p.add_argument("--ignore_files", dest="filesIgnore", type=str, default="",
                   help="Files of the data directory to ignore (comma separated)")
    p.add_argument("--decision", dest="decision", type=str, default=None,
                   help="Decision file rootname for the acquisition (default: SEG-Y headers)")
    # Data selection and source (makutu.forwardBackward.dataprocessing)
    p.add_argument("--offsetMin", dest="offsetMin", type=float, default=0.0, help="Smallest offset kept [m]")
    p.add_argument("--offsetMax", dest="offsetMax", type=float, default=float("inf"), help="Largest offset kept [m]")
    p.add_argument("--muteVelocity", dest="muteVelocity", type=float, default=None,
                   help="Top mute: samples before muteTime + offset / muteVelocity set to 0 [m/s]")
    p.add_argument("--muteTime", dest="muteTime", type=float, default=0.0, help="Mute time at zero offset [s]")
    p.add_argument("--timeWindow", dest="timeWindow", type=float, default=None,
                   help="Length of the window kept after the mute line [s] (default: to the end)")
    p.add_argument("--muteTaper", dest="muteTaper", type=float, default=0.05,
                   help="Cosine ramps of the mute and of the window [s]")
    p.add_argument("--estimateSource", dest="estimateSource", type=str2bool, default=False,
                   help="Estimate the source wavelet (least-squares filter) at each filter frequency")
    # Outputs
    p.add_argument("--trace_directory", dest="traceDir", type=str, default="seismicTrace",
                   help="Output directory of the simulated traces")
    p.add_argument("--exportSeismicTrace", dest="exportSeismicTrace", type=str2bool, default=False)
    p.add_argument("--exportResidual", dest="exportResidual", type=str2bool, default=False)
    p.add_argument("--pCostdir", dest="pCostDir", type=str, default="partialCostFunction")
    p.add_argument("--gradDir", dest="gradDir", type=str, default="partialGradient",
                   help="Directory of the full gradients in debug mode")
    # Inversion
    p.add_argument("--model", dest="model", type=str, choices=["c", "1/c", "1/c2"], default="c",
                   help="Parametrisation")
    p.add_argument("--method", dest="method", type=str, default="steepestDescent",
                   help="Optimisation method (steepestDescent, SD2, CGPR, CGPR2 or a scipy method)")
    p.add_argument("--gDn", dest="gDn", type=int, default=0,
                   help="Gradient normalisation: 0 none, 1 max, 2 norm (0 with SD2, CGPR and scipy)")
    # Gradient conditioning (makutu.imaging.GradientConditioner)
    p.add_argument("--minDepth", dest="minDepth", type=float, default=0.0,
                   help="Depth [m] above which the model is not updated")
    p.add_argument("--gradEdge", dest="gradEdge", type=float, default=0.0,
                   help="Band [m] along the sides and the bottom where the model is not updated")
    p.add_argument("--gradTaper", dest="gradTaper", type=float, default=0.0,
                   help="Width [m] of the cosine ramps of minDepth and gradEdge")
    p.add_argument("--illumination", dest="illumination", type=str2bool, default=False,
                   help="Divide the gradient by the illumination (pseudo-Hessian, source side)")
    p.add_argument("--illumEps", dest="illumEps", type=float, default=0.01,
                   help="Stabilisation of the illumination division, relative to its maximum")
    p.add_argument("--minimizerOptions", dest="miniOptions", type=str, default="{}",
                   help="Options of the optimiser as a Python dict; minModel and maxModel are "
                        "the velocity bounds [m/s] (default 500, 8500)")

    args, _ = p.parse_known_args(hdrList)
    return args


def build_solver(wf, vp_file):
    """
    Mesh (from the SEP model ``vp_file``) and initialised AcousticSolver for the
    parameters ``wf`` of :func:`parse_workflow_parameters`. Also used to compute the
    observed data of a synthetic case on the same mesh, time step and sponge.

    Returns
    -------
        (AcousticSolver, SimpleNamespace of makutu.mesh.mesh_from_sep)
    """
    m = mesh_from_sep(vp_file, wf.sourceFreq, wf.order, rho_file=wf.rhoFile, ppw=wf.ppw, cfl=wf.cfl,
                      elem_size=wf.elemSize, free_surface_on_top=wf.freeSurface)
    dt = wf.dt if wf.dt is not None else m.dt
    sponge = wf.sponge
    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            model_location=FTSolver.ModelLocationType.ONELEMENTS,
                            sponge_size=[sponge] * 3, surface_sponge=wf.freeSurface,
                            dt=dt, minTime=0.0, maxTime=wf.maxtime, dtSeismo=wf.dtSeismo or dt,
                            sourceType=wf.sourceType, sourceFreq=wf.sourceFreq, **m.mesh.solver_kwargs())
    solver.initialize()
    if wf.dtWaveField is not None:
        solver.snapshotStep = max(1, int(round(wf.dtWaveField / dt)))
    solver.snapshotCompression = None if wf.snapshotCompression == "none" else wf.snapshotCompression
    print(f"Time  : dt {dt * 1e3:.3f} ms, {solver.n_time_steps} steps, seismograms every "
          f"{solver.dtSeismoRecorded * 1e3:.3f} ms, sponge {sponge:g} m")
    step = getattr(solver, "snapshotStep", None) or solver.snapshotStepFor()
    print(f"Gradient: snapshot every {step} steps, {solver.snapshotMemory(step) / 2 ** 20:.0f} MiB of host memory "
          f"per shot ({wf.snapshotCompression})")
    return solver, m


def fullWaveformInversion(minimizer, x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=False,
                          traceDirectory=".", exportResidual=False, model="c", pCostDir="partialCostFunction",
                          dinfo="seismoTrace_shot;5;.sgy", gDn=0, freqList=("all",), firstFreq=0,
                          problem=None, estimateSource=False):
    """
    Minimisation for every filter frequency from ``freqList[firstFreq]``, each starting from
    the model of the previous one; with ``estimateSource``, the source wavelet of the problem
    is estimated first on the current model.

    Returns
    -------
        np.ndarray : final model
    """
    for ifreq, freq in enumerate(freqList):
        if ifreq < firstFreq:
            continue
        print("\n", "*" * 80, sep="")
        print(f"Filtering frequency = {freq}  [ {ifreq + 1} / {len(freqList)} ]")
        print("*" * 80, end="\n\n")
        if estimateSource:
            problem.solver.setModelForGradient(model)
            problem.solver.updateModel(x, problem.minModel, problem.maxModel)
            freqs, c = problem.estimateSource(dataDirectory, ifreq, dinfo=dinfo)
            band = np.abs(c) > 0.1 * np.abs(c).max()
            print(f"Source filter estimated: |c| {np.abs(c[band]).min():.3g} - {np.abs(c[band]).max():.3g} "
                  f"over {freqs[band].min():.3g} - {freqs[band].max():.3g} Hz\n")
        x = minimizer.optimize(x=x, args=(dataDirectory, gradDirectory, exportSeismicTrace, traceDirectory,
                                          exportResidual, ifreq, model, pCostDir, dinfo, gDn))
    return x


def main():
    args = parse_args()
    wf = parse_workflow_parameters(args.pfile)
    rank, nranks = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    if rank > 0:
        sys.stdout = open(os.devnull, "w")          # rank 0 prints (the ranks take the same steps)
    mode = "DBG" if (args.DBG or args.dbgIO) else "REL"
    exportSeismicTrace = True if args.dbgIO else wf.exportSeismicTrace
    exportResidual = True if args.dbgIO else wf.exportResidual
    freqFilterList = (["all"] if "all" in wf.freqFilterList
                      else np.asarray(wf.freqFilterList.split(","), dtype=float).tolist())

    # Model, mesh, solver
    if wf.vpFile is None:
        raise SystemExit("vpFile (initial model, SEP) is required: it also sets the mesh")
    solver, m = build_solver(wf, wf.vpFile)

    # Acquisition
    if wf.decision is not None:
        acquisition = DecisionAcquisition(decision_filename=wf.decision, scalco=1, scalel=1)
    else:
        ignore = [f for f in wf.filesIgnore.split(",") if f]
        acquisition = SEGYAcquisition(segdir=wf.dataDir, ignore=ignore)
    print(f"Shots : {len(acquisition.shots)}, over {nranks} MPI rank(s)")

    # Parametrisation and bounds
    options = eval(wf.miniOptions)
    vmin, vmax = options.pop("minModel", 500.0), options.pop("maxModel", 8500.0)
    to_param = {"c": lambda v: v, "1/c": lambda v: 1.0 / v, "1/c2": lambda v: 1.0 / v ** 2}[wf.model]
    minx, maxx = sorted((to_param(vmin), to_param(vmax)))

    if args.restart and args.rfile is not None:
        with h5py.File(args.rfile, "r") as f:
            x0 = f["velocity"][:]
        if x0.size != solver.getVelocityModel().size:
            raise SystemExit(f"{args.rfile}: {x0.size} values, the mesh has {solver.getVelocityModel().size} elements")
        if not 1 <= args.rfreq <= len(freqFilterList):
            raise SystemExit(f"--restart_freq {args.rfreq}: freqFilterList has {len(freqFilterList)} frequencies")
        print(f"Restart model: {args.rfile}, frequency {freqFilterList[args.rfreq - 1]} "
              f"[{args.rfreq} / {len(freqFilterList)}], next model file velocityModel{args.riter}.hdf5")
        firstFreq, iteration = args.rfreq - 1, args.riter
    else:
        x0 = to_param(solver.getVelocityModel().astype(np.float64))
        firstFreq, iteration = 0, 1
    selection = None
    if wf.offsetMin > 0 or np.isfinite(wf.offsetMax) or wf.muteVelocity is not None:
        selection = DataSelection(offsetMin=wf.offsetMin, offsetMax=wf.offsetMax, muteVelocity=wf.muteVelocity,
                                  muteTime=wf.muteTime, window=wf.timeWindow, taper=wf.muteTaper)
        print(selection)
    problem = FWIProblem(solver, acquisition, minModel=minx, maxModel=maxx, freqList=freqFilterList, mode=mode,
                         model=x0.copy(), iteration=iteration, dataSelection=selection)

    precond = None
    if wf.minDepth > 0 or wf.gradEdge > 0 or wf.illumination:
        precond = GradientConditioner(solver, minDepth=wf.minDepth, edge=wf.gradEdge, taper=wf.gradTaper,
                                      illumination=wf.illumination, eps=wf.illumEps, problem=problem)
        print(f"Gradient conditioning: minDepth {wf.minDepth:g} m, edge {wf.gradEdge:g} m, "
              f"taper {wf.gradTaper:g} m, illumination {wf.illumination} (eps {wf.illumEps:g})")
    minimizer = Minimizer(problem.func, method=wf.method, jac=problem.gradf, bounds=[[minx, maxx]] * len(x0),
                          tol=None, callback=problem.callback, options=options, precond=precond)
    x = fullWaveformInversion(minimizer, x0, wf.dataDir, gradDirectory=wf.gradDir,
                              exportSeismicTrace=exportSeismicTrace, traceDirectory=wf.traceDir,
                              exportResidual=exportResidual, model=wf.model, pCostDir=wf.pCostDir,
                              dinfo=wf.dinfo, gDn=wf.gDn, freqList=freqFilterList, firstFreq=firstFreq,
                              problem=problem, estimateSource=wf.estimateSource)
    if rank == 0:
        writeFields("finalModel.hdf5", {"velocity": x}, solver)
    print("Final model written to finalModel.hdf5")
    problem.release()             # every Kokkos view must be released before kokkos.finalize
    del solver, m, problem, minimizer, precond


if __name__ == "__main__":
    main()
    kokkos.finalize()
