import os
from copy import deepcopy
import numpy as np
import math

from utilities.acquisition import EQUISPACEDAcquisition
from utilities.imaging import gradientComputation, \
                              computeFullGradient
from utilities.forwardBackward import directSimulation, \
                                      computeFullCostFunction
from utilities.solvers import AcousticSolver
from utilities.imaging import setModelAsGlobal, initializeGlobalVariables
from utilities.imaging import globals as gb

import h5py
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():

    #Time Parameters
    minTime = 0.0
    maxTime = 0.8
    dt = 0.001
    dtSeismo = 0.001
    dtWaveField = 0.001

    # Parameters
    exportSeismicTrace = True
    exportResidual = False

    sourceType = "ricker2"
    sourceFreq = 2
    freqFilterList = ["all"]

    minVel = 1500
    maxVel = 4500
    minDepth = 50

    traceDirectory = "seismicTraceTestGradient"
    computeCostFunction = True
    gradDirectory = "partialGradient"
    dataDirectory = "seismicDataTestGradient"

    # Set Acquisition
    # sources
    startFirstSourceLine = [1000.1, 1000.1, 1000.1]
    endFirstSourceLine = [1000.1, 1000.1, 1000.1]
    numberOfSourceLines = 1
    sourcesPerLine = 1

    #receivers
    startFirstReceiversLine = [600.1, 1000.1, 1020.1]
    endFirstReceiversLine = [1400.1, 1000.1, 1020.1]
    numberOfReceiverLines = 1
    receiversPerLine = 50

    mode="DBG"
    model="c"
    random_model=False
    rng = np.random.default_rng()
    filter_bound=False

    acquisition = EQUISPACEDAcquisition(startFirstSourceLine=startFirstSourceLine,
                                        endFirstSourceLine=endFirstSourceLine,
                                        numberOfSourceLines=numberOfSourceLines,
                                        sourcesPerLine=sourcesPerLine,
                                        startFirstReceiversLine=startFirstReceiversLine,
                                        endFirstReceiversLine=endFirstReceiversLine,
                                        numberOfReceiverLines=numberOfReceiverLines,
                                        receiversPerLine=receiversPerLine,
                                        dt=dt)

    solver = AcousticSolver(dt,
                            minTime,
                            maxTime,
                            dtSeismo,
                            dtWaveField,
                            sourceType,
                            sourceFreq)

    listOfAcquisition = acquisition.splitAcquisition()

    initializeGlobalVariables(solver, acquisition, listOfAcquisition, minVel, maxVel, minDepth, freqFilterList,mode)

    n = 200
    if random_model :
        a = 100*rng.random((n,n,n))
    else:
        a = np.zeros((n,n,n))
    a[:,:,:] += 1500

    startModel = deepcopy(a.reshape(n*n*n))
    print("Searched inversion model min start:%E max start:%E"%(np.min(startModel),np.max(startModel)))

    startModelFile = "velocityModel.hdf5"
    f = h5py.File(startModelFile, 'w', driver="mpio", comm=comm)
    f.create_dataset("velocity", data=startModel[:], chunks=True, maxshape=(len(startModel),))
    f.close()

    setModelAsGlobal(startModel)


    for i in range(len(listOfAcquisition)):
        directSimulation(i, exportSeismicTrace, dataDirectory, startModelFile=startModelFile, model=model)

    if random_model :
        a = 100*rng.random((n,n,n))
    else:
        a = np.zeros((n,n,n))
    a[:,:,:] += 1800
    startModel = deepcopy(a.reshape(n*n*n))
    print("Initial inversion model min start:%E max start:%E"%(np.min(startModel),np.max(startModel)))
    f = h5py.File(startModelFile, 'w', driver="mpio", comm=comm)
    f.create_dataset("velocity", data=startModel[:], chunks=True, maxshape=(len(startModel),))
    f.close()

    setModelAsGlobal(startModel)

    print(rank,"Start gradientComputation")
    gradAM = np.zeros(n*n*n)
    cellsPerRank = int(n*n*n / size)

    for i in range(len(listOfAcquisition)):
        gradientComputation(i, dataDirectory, gradDirectory, exportSeismicTrace, traceDirectory, exportResidual, startModelFile=startModelFile,model=model,dinfo="seismoTrace_shot;5;_%d.sgy"%(sourceFreq),computeCF=True)
    computeFullGradient(gradDirectory)
    print(rank,"after computeFullGradient")

        print(rank,"reading file fullGradient")
        h5F = h5py.File("fullGradient.hdf5", 'r', driver="mpio",comm=comm)
        if rank == size - 1:
            gradAM[rank*cellsPerRank::] = h5F["fullGradient"][rank*cellsPerRank::]
        else:
            gradAM[rank*cellsPerRank:(rank+1)*cellsPerRank] = h5F["fullGradient"][rank*cellsPerRank:(rank+1)*cellsPerRank]

        print(rank,"before computeFullCostFunction",gb.totalShots)
        fullCostFunction = computeFullCostFunction(gb.totalShots)
        print(rank,"after computeFullCostFunction",fullCostFunction)

        f0 = deepcopy(fullCostFunction)

    if rank == 0:
        delta = 200
    else:
        delta = None

    delta = comm.bcast(delta, root = 0)
    mask = np.zeros((n,n,n))
    if filter_bound:
        #apply perturbation only at center
        mask[50:-50,50:-50,50:-50]=1
    else : 
        #apply perturbation exevywhere
        mask[:,:,:]=1
    maskf=deepcopy(mask.reshape(n*n*n))
        
    for i in range(20):
        startModel = deepcopy(a.reshape(n*n*n))

        h = 0.5**i

        comm.Barrier()

        startModel[:] += delta*h*maskf[:]
        print("min start:%E max start:%E"%(np.min(startModel),np.max(startModel)))
        f = h5py.File(startModelFile, 'w', driver="mpio", comm=comm)
        f.create_dataset("velocity", data=startModel[:], chunks=True, maxshape=(len(startModel),))
        f.close()

        gb.setModelAsGlobal(startModel)

        for j in range(len(listOfAcquisition)):
            directSimulation(j, exportSeismicTrace, traceDirectory, computeCostFunction, dataDirectory, startModelFile=startModelFile,model=model,dinfo="seismoTrace_shot;5;_%d.sgy"%(sourceFreq))
        fullCostFunction = computeFullCostFunction(gb.totalShots)

            f1 = deepcopy(fullCostFunction)
            df = f1 - f0

            err = abs(df - h*delta*sum(gradAM))
            if i==0:
                err0=err
                errm1=err
                hm1=h
            else:
                slopes=(math.log(abs(err))-math.log(abs(errm1)))/(math.log(abs(h))-math.log(abs(hm1)))
                errm1=err
                hm1=h

            if rank == 0:
                os.system("mkdir -p errGradient/")
                with open("errGradient/err.txt", 'a') as f:
                    f.write("h = " + str(h) + "\n")
                    f.write("err = " + str(err) + "\n\n")
                with open("errGradient/err_gnu.txt", 'a') as f:
                    f.write("%E %E\n"%(h,err))
                with open("errGradient/err_gnu.txt", 'a') as f:
                    f.write("%E %E\n"%(h,err))
                with open("errGradient/err_gnu_form.txt", 'a') as f:
                    f.write("%E %E %E\n"%(h,h**2,err/err0))
                if i==0:
                    with open("errGradient/cost.txt", 'a') as f:
                        f.write("ref %E\n"%(f0))
                with open("errGradient/cost.txt", 'a') as f:
                    f.write("%d %E\n"%(i,f1))
                    
                if i>0:
                  with open("errGradient/slopes.txt", 'a') as f:
                    f.write("%.5f\n"%slopes)

    #writing gnuplot script to visualize results
    with open('gnuplot_script_slopes.sh', 'w') as f:
      f.write('#!gnuplot\n')
      f.write('set logscale y 10\n')
      f.write('plot "errGradient/err_gnu_form.txt" using ($0):1 w linesp, "errGradient/err_gnu_form.txt" using ($0):2 w linesp, "errGradient/err_gnu_form.txt" using ($0):3 w linesp\n')
      f.write('pause -1 "Hit any key to continue"\n')

if __name__ == "__main__":
    main()
