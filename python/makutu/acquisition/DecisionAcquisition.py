import numpy as np
from re import findall
from copy import deepcopy

from .Shot import Source, SourceSet, Receiver, Dasgeometry, ReceiverSet, Shot
from .Acquisition import Acquisition
from .decision_reader import read_decision_file


class DecisionAcquisition(Acquisition):
    """
    Acquisition defined from a decision file 
    
    Attributes
    ---------
        mesh : StructuredMesh, CartesianMesh2D, or None
            Mesh
        limited_aperture : bool
            Full or limited aperture flag
        database_filename : str
            Database corresponding to the decision file. Default is identity
        decision_filename : str
            Decision file name
        shots : Shot
            Shot configuration
        numberOfSources : int
            Total number of sources in the decision file
    """
    def __init__(self,
                 mesh=None,
                 dt=None,
                 **kwargs,
                ):
        """
        Parameters
        ----------
            mesh : StructuredMesh, CartesianMesh2D, or None
                Mesh of the problem (optional).
            dt : float
                Timestep
            **kwargs : decision_filename : str
                        Decision file name
                       database_filename : str
                        Database corresponding to the decision file
        """
        self.database = kwargs.get("database_filename", "identity")
        self.filename = kwargs.get("decision_filename", None)
        super().__init__(mesh, dt, **kwargs)
        self.type = "decisionAcquisition"


    def acquisition(self, **kwargs):
        """
        Set the shots configuration
        Parameters
        ----------
            **kwargs : keyword argument to be used in self.decisionReader()
        """
        kwargs = {"sorted": True, **kwargs}   # receivers in file order unless sorted=False
        srcPos, rcvPos, self.numberOfSources = self.decisionReader(self.filename, self.database, **kwargs)
        
        srcID = []
        for k in srcPos.keys():
                srcID += [int(findall("[0-9]+",k)[0])]

        shots = []
        for i in srcID:
            # 1 Shot = 1 source
            sourceSet = SourceSet()
            shotId = f"{i:05d}"

            sourceSet.append(Source(*srcPos[f"src{i}"]))
            receiverSet = ReceiverSet()
            for j in range(len(rcvPos[f"rcv_{i}"])):
                receiverSet.append(Receiver(*rcvPos[f"rcv_{i}"][j][0:3]))

            if 'dwords' in kwargs.keys():
                dasgeometrySet = ReceiverSet()
                for j in range(len(rcvPos[f"rcv_{i}"])):
                    dasgeometrySet.appendDAS(Dasgeometry(*rcvPos[f"rcv_{i}"][j][3:6]))

                shot = Shot(sourceSet, receiverSet, shotId, dasgeometrySet=dasgeometrySet)
            else:
                shot = Shot(sourceSet, receiverSet, shotId)
            shots.append(deepcopy(shot))
        self.shots = shots


    @staticmethod
    def decisionReader(decisionFile, databaseFile="identity", needTraces=False, zopp=False, **kwargs):
        """
        Retrieve the source and receiver data and traces from a decision file.

        Delegates to :func:`makutu.acquisition.decision_reader.read_decision_file`.

        Parameters
        ----------
            decisionFile : str
                Rootname of decision file
            databaseFile : str, optional
                Name of database file. Default is ``"identity"``.
            needTraces : bool, optional
                Whether to extract seismic trace data. Default is ``False``.
            zopp : bool
                Flag to flip Z coordinate sign. Default is ``False``.
            **kwargs :
                Forwarded to :func:`read_decision_file` (scalco, scalel, utmxy2xy,
                xy2utmxy, swords, rwords, dwords, gather, sorted).

        Returns
        -------
            d_src : dict
            d_rcv : dict
            n : int
        """
        return read_decision_file(
            decisionFile,
            database_file=databaseFile,
            need_traces=needTraces,
            zopp=zopp,
            **kwargs,
        )
