import numpy as np

from makutu.acquisition import Acquisition
from makutu.solvers import AcousticSolver
from makutu.output import SEPTraceOutput

from mpi4py import MPI


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Definition of the acquisition
    sources   = [[pos, 1000.1, 520.1] for pos in np.linspace(700.1, 1300.1, 5)]
    receivers = [[pos, 1000.1, 520.1] for pos in np.linspace(100.0, 1900.1, 80)]

    acquisition = Acquisition(sources=sources, receivers=receivers)

    # Solver definition
    dt = 0.001
    tmax = 1.
    solver = AcousticSolver(dt=dt,
                            maxTime=tmax)

    for ishot, shot in enumerate(acquisition.shots):

        solver.resetWaveField()
        solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())

        # Time loop
        for cycle in range(solver.n_time_steps):
            if rank == 0 and cycle % 100 == 0:
                print(f"step = {cycle} / {solver.n_time_steps}")
            solver.execute(cycle)

        shot.flag = "Done"
        if rank == 0 :
            print(f"Shot {shot.id} done")

        
        seismos = solver.getPressureAtReceivers()
    
        # Save the seismic traces
        directory = './seismicTraces/'
        rootname = f"seismo_{shot.id}"

        seismosOut = SEPTraceOutput(seismos, directory=directory, rootname=rootname)
        seismosOut.gather(verbose=True)
        seismosOut.export(dt=solver.dtSeismo)

        solver.resetWaveField()
        solver.resetPressureAtReceivers()