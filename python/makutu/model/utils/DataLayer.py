import numpy as np


class DataLayer:
    """
    Class containing methods to generate a homogeneous data layer

    Attributes
    -----------
        numberOfCells : tuple of len 3
            Number of cells in each dimension
        value : float
            Data value
        name : str
            Dataset name
        data : array of shape `numberOfCells`
            Array containing the data generated
            
    Examples
    ---------
        >>>> from makutu.model import DataLayer
        >>>> layer = DataLayer(numberOfCells=(10,2,3), value=1500, name="mediumVelocity")
        >>>> layer
        Dataset `mediumVelocity`
                Number of cells : (10, 2, 3)
                Value : 1500
    """
    def __init__(self, numberOfCells, value, name):
        """
        Parameters
            numberOfCells : len 3 array-like
                Number of cells in each dimension
            value : float
                Filling data value
            name : str
                Dataset name
        """
        self.numberOfCells = tuple(numberOfCells)
        
        self.name = name
        
        if isinstance(value, list):
            self.value = value[0]
        else:
            self.value = value
    
        self.__generateLayer()


    def __generateLayer(self):
        """
        Generate the data layer with the correct number of cells and filling value
        """
        self.data = np.full(self.numberOfCells, fill_value=self.value, dtype=float)


    def __repr__(self):
        return f'Dataset `{self.name}` \n\tNumber of cells : {self.numberOfCells} \n\tValue : {self.value}\n'



class BilayerData:
    """
    Class containing method to generate a bilayer data model

    Attributes
    -----------
        numberOfCells : tuple of len 3
            Number of cells in each dimension
        axis : int
            Axis of symmetry
        name : str
            Dataset name
        n1 : tuple of len 3
            Number of cells in each dimension for the layer 1
        n2 : tuple of len 3
            Number of cells in each dimension for the layer 2
        dataset1 : DataLayer
            Layer 1
        dataset2 : DataLayer
            Layer 2
        values : len 2 array-like
            Filling values of dataset 1 and 2
        data : array-like of shape `numberOfCells`
            Full model

    Examples
    ---------
        Bilayer with y axis of symmetry
        >>>> from makutu.model import BilayerData
        >>>> bilayerY = BilayerData(numberOfCells=(10,8,2,10), value=(1500,2500), name="velocity", axis=2)
        >>>> bilayerY
        Bilayer dataset `velocity`
                Layer 1, number of cells : (10, 8, 10)
                Layer 2, number of cells : (10, 2, 10)

        Bilayer with z axis of symmetry
        >>>> bilayerZ = BilayerData(numberOfCells=(10,10,8,2), value=(1500,2500), name="velocity")
        >>>> bilayerZ
        Bilayer dataset `velocity`
                Layer 1, number of cells : (10, 10, 8)
                Layer 2, number of cells : (10, 10, 2)

        Bilayer with x axis of symmetry
        >>>> bilayerX = BilayerData(numberOfCells=(8,2,10,10), value=(1500,2500), name="velocity", axis=1)
        >>>> bilayerX
        Bilayer dataset `velocity`
                Layer 1, number of cells : (8, 10, 10)
                Layer 2, number of cells : (2, 10, 10)
    """
    def __init__(self, numberOfCells, value, name, axis=3):
        """
        Parameters
        -----------
            numberOfCells : len 4 array-like
                Number of cells in each dimension \
            value : len 2 array-like
                Filling data values for each layer
            name : str
                Dataset name
            axis : int, optional
                Axis of symmetry \
                Default is 3 (z)   
        """
        self.axis = axis
        self.values = tuple(value)
        self.name = name

        self.__setNumberOfElements(numberOfCells)
        self.__generateData()


    def __setNumberOfElements(self, numberOfCells):
        """
        Set the number of elements in each dimensions

        Parameters
        -----------
            numberOfCells : len 4 array-like
                Number of cells in each dimensions
        """
        assert len(numberOfCells) == 4, "Require 4 numbers"
        assert self.axis in range(1,4)

        if self.axis == 1:
            self.n1 = (numberOfCells[0], numberOfCells[2], numberOfCells[3])
            self.n2 = (numberOfCells[1], numberOfCells[2], numberOfCells[3])

            self.numberOfCells = (numberOfCells[0] + numberOfCells[1], numberOfCells[2], numberOfCells[3])

        elif self.axis == 2:
            self.n1 = (numberOfCells[0], numberOfCells[1], numberOfCells[3])
            self.n2 = (numberOfCells[0], numberOfCells[2], numberOfCells[3])

            self.numberOfCells = (numberOfCells[0], numberOfCells[1] + numberOfCells[2], numberOfCells[3])
        
        else:
            self.n1 = (numberOfCells[0], numberOfCells[1], numberOfCells[2])
            self.n2 = (numberOfCells[0], numberOfCells[1], numberOfCells[3])

            self.numberOfCells = (numberOfCells[0], numberOfCells[1], numberOfCells[2] + numberOfCells[3])


    def __generateData(self):
        """
        Generate the two datasets
        """
        assert None not in self.n1 and None not in self.n2

        self.layer1 = DataLayer(self.n1, self.values[0], name="layer1")
        self.layer2 = DataLayer(self.n2, self.values[1], name="layer2")

        self.data = np.concatenate((self.layer1.data, self.layer2.data), axis=self.axis-1)


    def __repr__(self):
        return f'Bilayer dataset `{self.name}`\n\tLayer 1, number of cells : {self.n1}\n\tLayer 2, number of cells : {self.n2}\n'