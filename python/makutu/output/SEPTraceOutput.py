import os
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

from ..model.utils.SepModel import SEPModel

class SEPTraceOutput:
    """
    Class containing methods specifics to SEP seismic trace output

    Attributes
    -----------
        format : str
            Output filename extension
        directory : str
            Output directory \
            Default is current dir
        rootname : str
            Output root filename
        head : str
            Filename of the header
        data : array-like
            Seismic traces to export
        time : array-like
            Time corresponding to seismic traces
        gdata : array-like
            Gathered data
    """
    def __init__(self, seismo, rootname="seismoTrace_shot", directory="./", tIncluded=True, **kwargs):
        """
        Parameters
        ----------
            seismo : array-like
                Seismic traces to export
            rootname : str, optional
                Output root filename \
                Default is `seismoTrace_shot`
            directory : str, optional
                Output directory \
                Default is current dir
            tIncluded : bool, optional
                Whether time is included in seismos \
                Default is True
        """
        self.format = ".H"
        self.directory = directory
        self.rootname = rootname
        self.head = os.path.join( self.directory, self.rootname + self.format )

        if tIncluded:
            self.data, self.time = seismo[:, :-1], seismo[:, -1]
        else:
            self.data = seismo
            self.time = None

        self.gdata = None


    def gather(self, comm=pkl5.Intracomm(MPI.COMM_WORLD), root=None, verbose=False):
        """
        Gather the seismic traces on the root rank

        Parameters
        -----------
            comm : MPI communicator
                MPI communicator \
                Default is MPI.COMM_WORLD
            root : int
                Root rank \
                Default is 0
            verbose : bool \
                Print the min and max values of gathered data
        """
        if root is None:
            root = 0

        rank = comm.Get_rank()
        if rank == root:
            print("Gathering the seismo traces")
        
        n1 = self.data.shape[0]
        n3 = self.data.shape[1]

        if n3 == 0:
            if rank == root: print("No receivers found, output cancelled")
            return
 
        # Send seismos to rank 0
        gseismo = np.zeros( ( n1, n3 ) )
        gseismo[:,:] = self.data[:,:]

        if rank != root:
            comm.Send( gseismo, dest=root, tag=1 )
        else:
            tmp = np.zeros( ( n1, n3 ) )
            for r in range( comm.Get_size() ):
                if r != root:
                    comm.Recv( tmp, r, 1 )
                    for j in range( 0, n3 ):
                        if np.abs( np.min( gseismo[:,j] )) < 1e-7 and np.abs( np.max( gseismo[:,j] )) < 1e-8:
                            gseismo[:,j] += tmp[:,j]
        comm.Barrier()

        self.gdata = gseismo

        if verbose and rank == root:
            print(f"Sismos Min : {self.gdata.min()} - Max : {self.gdata.max()}")


    def export(self, dt=None, comm=pkl5.Intracomm(MPI.COMM_WORLD), **kwargs):
        """
        Export the sismo traces in .H file

        Parameters
        ----------
            dt : float, optional
                Seismo time step \
                If None (default), \
                    time step from GEOS seismos
            comm : MPI communicator, optional
                MPI communicator \
                Default is MPI.COMM_WORLD
        """
        rank = comm.Get_rank()
        root = kwargs.get( "root", 0 )

        if self.gdata is None:
            verbose = kwargs.get( "verbose", False )
            self.gather( comm=comm, root=root, verbose=verbose )

            if self.gdata is None:
                return

        if rank == root:
            print( f"Writing SEP file : {self.head}" )

            n1 = self.gdata.shape[0]
            n2 = 1
            n3 = self.gdata.shape[1]

            if dt is None and self.time is not None:
                dt = self.time[1] - self.time[0]
            elif dt is None and self.time is None:
                raise ValueError( "Timestep `dt` required for seismic traces output" )

            _, head = os.path.split( self.head )
            dictSEP = {
                "head": head, "bin": head + "@",
                "label1": "\"TIME\"", "label2": "\"CDP\"", "label3": "\"LINE\"",
                "o1": 0.0, "o2": 0.0, "o3" : 0.0,
                "d1": dt, "d2": 1, "d3": 1,
                "n1": n1, "n2": n2, "n3" : n3,
                "data_format": "native_float",
            }
            sepmodel = SEPModel( header=dictSEP, data=np.transpose( self.gdata ) )
            os.makedirs( self.directory, exist_ok=True )

            sepmodel.export( directory=self.directory )
