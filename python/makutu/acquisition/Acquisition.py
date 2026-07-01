import os
from copy import deepcopy
import numpy as np

from .Shot import Source, SourceSet, Receiver, Dasgeometry, ReceiverSet, Shot
from ..mesh.InternalMesh import StructuredMesh


class Acquisition:
    def __init__(self,
                 mesh=None,
                 dt=None,
                 **kwargs):
        """
        Parameters
        ----------
            mesh : StructuredMesh, or None
                Mesh of the problem. Required for limited-aperture and gradient
                workflows; optional for pure shot-loop workflows.
            dt : float
                Timestep
            kwargs : keyword arguments
            sources : list of list of float
                List of all sources coordinates (required)
            receivers : list of list of float
                List of all receivers coordinates (required)
            dasgeometry : list of list of float (optional)
                List of all DAS geometry information (to activate it, use dwords)
            acqId : int
                Acquisition id
                Default is 1
        """
        self.type = "acquisition"

        self.mesh = mesh
        self.limited_aperture = False

        self.acquisition(**kwargs)

        acqId = kwargs.get("acqId", 1)
        self.id = f"{acqId:05d}"

        self.dt = dt
        for shot in self.shots:
            if dt is not None:
                shot.dt = dt

            shot.setMesh(self.mesh)
         

    def loadMesh(self):
        """Load the mesh to set its properties (bounds, number of points, ...)"""
        if self.mesh is not None and not self.mesh.isSet:
            self.mesh.setMeshProperties()


    def getMesh(self):
        """
        Get the mesh associated to the acquisition

        Returns
        --------
            Mesh
                Mesh associated
        """
        return self.mesh


    def acquisition(self,
                    sources=None,
                    receivers=None,
                    **kwargs):
        """
        Set the shots configurations
        The same set of receivers is used for all shots

        Please provide the same type of variable for `sources` and `receivers`

        Parameters
        -----------
            sources : list of list of float or str
                Sources coordinates \
                If `sources` is str, filename containing all sources coordinates
            receivers : list of list of float
                Receivers coordinates
                If `receivers`
            
            If one wish to use DAS, it is necessary to parse the `dwords` parameter, even if it is None.

        Examples
        ---------
            >>> srcList = [[1,2,3],[4,5,6]]
            >>> rcvList = [[7,8,9],[10,11,12],[13,14,15]]
            >>> acquisition = Acquisition(sources=srcList, receivers=rcvList)
            >>> dasList = [[1,0.1,9],[2,0.4,12],[3,0.5,15]]
            >>> acquisition = Acquisition(sources=srcList, receivers=rcvList, dwords=None, dasgeometry=dasList)
        """
        if 'dwords' in kwargs.keys():
            dasgeometry = kwargs.get('dasgeometry', None)
            if sources is None or receivers is None or dasgeometry is None:
                raise ValueError(
                    "sources, receivers, and dasgeometry must all be provided "
                    "when using DAS (dwords)."
                )
            elif isinstance(sources, str) and isinstance(receivers, str) and isinstance(dasgeometry, str):
                sources = np.loadtxt(sources)
                receivers = np.loadtxt(receivers)
                dasgeometry = np.loadtxt(dasgeometry)

            numberOfDasgeometry = len(dasgeometry)
            dasgeometrySet = ReceiverSet([Dasgeometry(*dasgeometry[i])
                                   for i in range(numberOfDasgeometry)])
        else:
            if sources is None or receivers is None:
                raise ValueError(
                    "sources and receivers must be provided explicitly."
                )
            elif isinstance(sources, str) and isinstance(receivers, str):
                sources = np.loadtxt(sources)
                receivers = np.loadtxt(receivers)

        numberOfReceivers = len(receivers)
        numberOfSources = len(sources)

        receiverSet = ReceiverSet([Receiver(*receivers[i])
                                   for i in range(numberOfReceivers)])

        shots = []

        for i in range(numberOfSources):
            sourceSet = SourceSet()  #1 source per shot
            shot_id = f"{i+1:05d}"
            sourceSet.append(Source(*sources[i]))

            if 'dwords' in kwargs.keys():
                shot = Shot(sourceSet, receiverSet, shot_id, dasgeometrySet=dasgeometrySet)
            else:
                shot = Shot(sourceSet, receiverSet, shot_id)
            shots.append(deepcopy(shot))

        self.shots = shots


    def getSourceCenter(self):
        """
        Return the central position of the all the sources contained in the acquisition

        Returns
        -------
            3d list : Coordinates of the center
        """
        sourceSet = SourceSet()
        for shot in self.shots:
            sourceSet.appendSet(shot.getSourceList())

        center = sourceSet.getCenter()
        return center


    def limitedAperture(self, dist1=None, dist2=None, dist3=None, comm=None, export=True):
        """ 
        Redefine each shot mesh to correspond to a limited aperture configuration.

        Parameters
        ---------
            mesh : StructuredMesh    
                Original mesh
            dist1 : float
                Distance to the submesh center in the 1st direction
            dist2 : float
                Distance to the submesh center in the 2nd direction
            dist3 : float
                Distance to the submesh center in the 3rd direction
            comm : MPI.COMM_WORLD
                MPI communicator
        """
        if isinstance(self.mesh, StructuredMesh):
            print("WARNING:\n")
            print("Limited Aperture configuration not handled yet for Structured Mesh.\n")


    def setMesh(self, mesh):
        """Set the mesh associated to the acquisition"""
        self.mesh = mesh


    def splitAcquisition(self,**kwargs):
        """
        Split the shots such that one Acquisition = 1 Shot

        Returns
        --------
            listOfAcquisition : list
                list of Acquisition objects such that 1 Shot = 1 Acquisition
            
            If using DAS, the dwords parameter must be parsed, even if it is None
        """
        listOfAcquisition = []
        for shot in self.shots:
            if 'dwords' in kwargs.keys():
                a = Acquisition(mesh=self.mesh, sources=shot.getSourceCoords(), receivers=shot.getReceiverCoords(), dasgeometry=shot.getDASGeometry(), dt=shot.dt)
            else:
                a = Acquisition(mesh=self.mesh, sources=shot.getSourceCoords(), receivers=shot.getReceiverCoords(), dt=shot.dt)
            a.shots[0].id = shot.id
            a.shots[0].dt = shot.dt

            listOfAcquisition.append(a)

        return listOfAcquisition
