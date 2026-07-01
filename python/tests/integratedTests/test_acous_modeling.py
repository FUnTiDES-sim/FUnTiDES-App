import numpy as np
import pytest

from utilities.acquisition import Acquisition, EQUISPACEDAcquisition
from utilities.solvers import AcousticSolver
from utilities.output import SEPTraceOutput

from mpi4py import MPI


#Type of Acquisition
@pytest.fixture
def acquisition(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def equiAcquisition():
    startFirstSourceLine = [100.1, 250.1, 20.1]
    endFirstSourceLine = [300.1, 500.1, 20.1]
    numberOfSourceLines = 1
    sourcesPerLine = 2
    startFirstReceiversLine = [20.1, 250.1, 30.1]
    endFirstReceiversLine = [20.1, 500.1, 30.1]
    startLastReceiversLine = [400.1, 20.1, 30.1]
    endLastReceiversLine = [400.1, 500.1, 30.1]
    numberOfReceiverLines = 30
    receiversPerLine = 10

    return EQUISPACEDAcquisition(startFirstSourceLine=startFirstSourceLine,
                                endFirstSourceLine=endFirstSourceLine,
                                numberOfSourceLines=numberOfSourceLines,
                                sourcesPerLine=sourcesPerLine,
                                startFirstReceiversLine=startFirstReceiversLine,
                                endFirstReceiversLine=endFirstReceiversLine,
                                startLastReceiversLine=startLastReceiversLine,
                                endLastReceiversLine=endLastReceiversLine,
                                numberOfReceiverLines=numberOfReceiverLines,
                                receiversPerLine=receiversPerLine)


@pytest.fixture
def simpleAcquisition():
    sources =  [[f"{srcX:.1f}", "250.1", "20.1"] for srcX in np.linspace(100.1, 600.1, 3)]
    receivers = [[f"{rcvX:.1f}", "250.1", "30.1"] for rcvX in np.linspace(20.1, 489.1, 300)]
    acquisition = Acquisition(sources=sources, receivers=receivers)
    return acquisition


class TestModelingWorkflow:
    def run(self, acquisition):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Acquisition set-up
        dt = .001
        tmax = 1.
        dtSeismoTrace= .05

        # Propagation Forward
        solver = AcousticSolver(dt=dt, maxTime=tmax, dtSeismo=dtSeismoTrace)

        for ishot, shot in enumerate(acquisition.shots):
            solver.resetWaveField()
            solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())

            t = 0
            cycle = 0                
            while t < tmax :
                if rank == 0 and cycle%100 == 0:
                    print(f"time = {t:.3f}s, dt = {dt:.4f}, iter = {cycle+1}")
                solver.execute(t)
                t += dt
                cycle += 1
            sismos = solver.getPressureAtReceivers()

            seismoOut = SEPTraceOutput(sismos, format="SEP")
            seismoOut.gather(verbose=True)

            solver.reinitSolver()
        return 0 
   
    @pytest.mark.parametrize('acquisition', ['equiAcquisition', 'simpleAcquisition'], indirect=True)
    def test_AcousticRun(self, acquisition):
        assert self.run(acquisition) == 0
