import numpy as np
import argparse
import h5py

from utilities.acquisition import SEGYAcquisition
from utilities.forwardBackward import func, \
                                      callback
from utilities.imaging import initializeGlobalVariables, \
                              setModelAsGlobal, \
                              gradf, \
                              Minimizer
from utilities.imaging import globals as gb
from utilities.solvers import AcousticSolver

from mpi4py import MPI
from mpi4py.util import pkl5

comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()


def parse_args():
    parser = argparse.ArgumentParser("FWI workflow parser")
    parser.add_argument("--param_file", type=str, dest="pfile",
                        help="File containing all the parameters necessary")
    parser.add_argument("--r", "--restart", action="store_true", default=False,
                        dest="restart",
                        help="Flag to restart from existing model file")
    parser.add_argument("--rfile", "--restart_model_file", 
                        type=str, dest="rfile", default=None,
                        help="HDF5 file containing the model for restart. WARNING: the restart model (c, 1/c or 1/c2) used should be the same as the one requested for the current run")
    parser.add_argument("--model_name", "--mname", dest="mname",
                        type=str, default="mediumVelocity",
                        help="Velocity model field name in file")
    parser.add_argument("--dbg", dest="DBG",
                        action="store_true", default=False,
                        help="Debug mode\
                              Save all intermediate full gradient \
                              Save all intermediate full cost function \
                              Don't remove the partial cost functions files \
                              Don't remove the partial gradient directories \
                              Default is False")
    parser.add_argument("--dbg+", dest="dbgIO",
                        action="store_true", default=False,
                        help="Mode Debug+, \
                              Activate the debug mode, \
                              Save all the seismic traces, \
                              Save all the residual \
                              Default is False.")

    args, _ = parser.parse_known_args()
    return args


def parse_workflow_parameters(pfile):
    with open(pfile, "r") as f:
        hdrStr = f.read()

    hdrList = []
    for fl in hdrStr.split('\n'):
        l = fl.split("#")[0]
        if l:
            # add "--" to facilitate parsing that follows
            l = "--" + l
            hdrList += l.split("=")
    
    parser = argparse.ArgumentParser("FWI workflow parser")
    parser.add_argument("--mintime", dest="mintime",
                        default=None, type=float,
                        help="Min time for the simulation")
    parser.add_argument("--maxtime", dest="maxtime",
                        default=None, type=float,
                        help="Max time for the simulation")
    parser.add_argument("--dt", dest="dt",
                        default=None, type=float,
                        help="Time step of simulation")
    parser.add_argument("--dtSeismo", dest="dtSeismo",
                        default=None, type=float,
                        help="Time step for ")
    parser.add_argument("--dtWaveField", dest="dtWaveField",
                        required=True, type=float,
                        help="Time step for the wavefields. \
                         Required for a FWI calculation")
    parser.add_argument("--dataDirectory", dest="dataDir",
                        type=str, default="seismicTrace",
                        help="Directory containing data for residual/cost function" )
    parser.add_argument("--ignore_files", dest="filesIgnore",
                        type=str, default="",
                        help="Files to ignore in data directory")
    parser.add_argument("--dinfo", dest="dinfo",
                        type=str, default="seismoTrace_shot;5;.sgy",
                        help="Rootname of data traces, shot id formatting (e.g. 000123 is 6), extension, separated by `;`\
                        Default is `seismoTrace_shot;5;.sgy`")
    parser.add_argument("--trace_directory", dest="traceDir",
                        type=str, default="seismicTrace",
                        help="Seismo trace output directory")
    parser.add_argument("--exportSeismicTrace", dest="exportSeismicTrace",
                        action="store_true", default=False,
                        help="Flag to export the seismo traces. \
                            Default is false")
    parser.add_argument("--exportResidual", dest="exportResidual",
                        action="store_true", default=False,
                        help="Flag to export the residual. \
                            Default is false")
    parser.add_argument("--pCostdir", dest="pCostDir",
                        type=str, default="partialCostFunction",
                        help="Partial cost function folder name")
    parser.add_argument("--sourceType", dest="sourceType",
                        type=str,
                        help="Source type")
    parser.add_argument("--sourceFreq", dest="sourceFreq",
                        type=float, 
                        help="Source frequency")
    parser.add_argument("--freqFilterList", dest="freqFilterList",
                        type=str, default="all",
                        help="Frequency filter list. \
                              Default is 'all'. \
                              Format: fq1,fq2,fq3")
    parser.add_argument("--gradDir", dest="gradDir",
                        type=str, default="partialGradient",
                        help="Directory to store the gradients")
    parser.add_argument("--gDn", dest="gDn",
                        type=int, default=0,
                        help="Option to normalize gradient")
    parser.add_argument("--minDepth", dest="minDepth",
                        type=float, default=0,
                        help="Minimal depth")
    parser.add_argument("--model", dest="model",
                        type=str, choices=["c", "1/c", "1/c2"],
                        help="Model considered")
    parser.add_argument("--method", dest="method",
                        type=str, default="steepestDescent",
                        help="Optimization method")
    parser.add_argument("--minimizerOptions", dest="miniOptions",
                        type=str, default="{}",
                        help="Options for the FWI minimizer")

    args, _ = parser.parse_known_args(hdrList)
    return args
        

def fullWaveformInversion( minimizer, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=False, traceDirectory=".", exportResidual=False, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy",gDn=0 ):
    """Perform FWI simulation

    Parameters:
    -----------
        minimizer : Minimizer
            Optimization method
        dataDirectory : str
            Directory for residual/costFunction computation.
        gradDirectory : str
            Directory where to output the partial gradients. Default creates (if not already exists) a new directory 'partialGradient'
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy. If not set default is True
        traceDirectory : str
            Directory where to export the seismic traces. If not set default is current directory
        exportResidual : bool
            Wheter or not to export the residual if computeCostFunction is set to True. Default is False
        model : str
            Model considered
        pCostDir : str
            Partial cost function directory
    """
    for ifreq, freq in enumerate(gb.freqList):
            print("\n", "*"*80, sep="")
            print(f"Filtering frequency = {freq}  [ {ifreq + 1} / {len(gb.freqList)} ]" )
            print("*"*80, end="\n\n")

            minimizer.optimize(x=gb.model,
                               args=(dataDirectory,
                                     gradDirectory,
                                     exportSeismicTrace,
                                     traceDirectory,
                                     exportResidual,
                                     ifreq,
                                     model,
                                     pCostDir,
                                     dinfo,
                                     gDn)
            )


if __name__ == "__main__":

    args = parse_args()

    wf_args = parse_workflow_parameters(args.pfile)

    mode = "DBG" if ( args.DBG or args.dbgIO ) else "REL"

    #Time Parameters
    minTime = wf_args.mintime
    maxTime = wf_args.maxtime
    dt = wf_args.dt
    dtSeismo = wf_args.dtSeismo
    dtWaveField = wf_args.dtWaveField

    #Seismic trace parameters
    traceDir = wf_args.traceDir

    if args.dbgIO:
        exportSeismicTrace = True
        exportResidual = True
    else:
        exportSeismicTrace = wf_args.exportSeismicTrace
        exportResidual = wf_args.exportResidual

    #Forward/Backward computation parameters
    sourceType = wf_args.sourceType
    sourceFreq = wf_args.sourceFreq
    if "all" not in wf_args.freqFilterList:
        freqFilterList = np.asarray(wf_args.freqFilterList.split(","), dtype=float).tolist()
    else:
        freqFilterList = ["all"]
   
 
    #fwi parameters
    method = wf_args.method
    gradDir = wf_args.gradDir
    dataDir = wf_args.dataDir
    filesToIgnore = wf_args.filesIgnore.split(",")
    dinfo = wf_args.dinfo
    gDn = wf_args.gDn
    
    model = wf_args.model

    minimizerOptions = eval(wf_args.miniOptions)

    #tol = minimizerOptions.pop("tol",0.0001)
    tol = None
    #Jtol = minimizerOptions.pop("Jtol",0.0001)
    
    # Set model bound constraints
    minx = minimizerOptions.pop("minModel", 500)
    maxx = minimizerOptions.pop("maxModel", 8500)
    
    if model == "1/c2":
        minx, maxx = 1 / maxx**2, 1 / minx**2
    elif model == "1/c":
        minx, maxx = 1 / maxx, 1 / minx

    minDepth = wf_args.minDepth
    
    pCostDir = wf_args.pCostDir

    #Set Acquisition
    acquisition = SEGYAcquisition(segdir=wf_args.dataDir,
                                  dt=dt,
                                  ignore=filesToIgnore
                                  )


    print(f"\n Restart model file is {args.rfile} \n")
    if args.restart and args.rfile is not None:
        startModelFile = args.rfile
        f = h5py.File(startModelFile, 'r', driver="mpio", comm=comm)
        x = f["velocity"][:]
        f.close()
        startModel = x
    else:
        startModelFile = None
        x = acquisition.getMesh().getArray(args.mname, copy=True)
        
        if model == "1/c2":
            startModel = 1/(x*x)         
        elif model == "1/c":
            startModel = 1/x
        else:
            startModel = x

    setModelAsGlobal(startModel)

    solver = AcousticSolver(dt,
                            minTime,
                            maxTime,
                            dtSeismo,
                            dtWaveField,
                            sourceType,
                            sourceFreq,
                            )

    # Solver with Caliper output
    # solver = AcousticSolver(dt,
    #                         minTime,
    #                         maxTime,
    #                         dtSeismo,
    #                         dtWaveField,
    #                         sourceType,
    #                         sourceFreq)
   
    minimizer = Minimizer(func,
                          method=method,
                          jac=gradf,
                          bounds=[[minx,maxx]]*len(startModel),
                          tol=tol,
                          callback=callback,
                          options=minimizerOptions)

    
    listOfAcquisition = acquisition.splitAcquisition()
    initializeGlobalVariables(solver, acquisition, listOfAcquisition, minx, maxx, minDepth, freqFilterList, mode)

    fullWaveformInversion(gradDirectory=gradDir, exportSeismicTrace=exportSeismicTrace, traceDirectory=traceDir, dataDirectory=dataDir, exportResidual=exportResidual, minimizer=minimizer, model=model, pCostDir=pCostDir, dinfo=dinfo, gDn=gDn)
