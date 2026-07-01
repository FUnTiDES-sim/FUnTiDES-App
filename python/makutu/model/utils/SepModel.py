import os
import sys
import numpy as np
import argparse
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

class SEPModel:
    """
    Define a SEP model
        
    Attributes  
    ----------
        header : SEPHeader
            Object containing SEP header information
        bin : SEPBin
            Object containing SEP data
    """
    def __init__(self, header=None, data=None, name=None):
        """
        Parameters
        ----------
            header: str or dict
                SEP Header. Header filename if str, \
                    dict containing SEP header informations if dict
        """
        self.type = "SEP"
        self.header = SEPHeader(header)
        self.bin = SEPBin(header=self.header, data=data)
        self.name = name


    def getHeaderInfos(self):
        """Print header properties"""
        print(self.header)


    def getBinInfos(self):
        """Print data (binary) properties"""
        print(self.bin)

    
    def getModel(self, datatype=None):
        """
        Return the cells or points model

        Parameters
        ----------
            datatype : str
                "cells" or "points" type model
        Returns
            numpy array
        """
        return self.bin.getModel(datatype)
        

    def createAllBlocks(self, nb, bext="", verbose=False, comm=None):
        """
        Partition the original SEP model into blocks and export
        
        Parameters
        ----------
            nb : 3d array-like
                Number of blocks for each dimension
            bext : str
                Suffix to form block filenames
            verbose : bool
                Print block information or not
            comm : MPI communicator
                MPI communicator
        """
        if comm is None:
            #comm = MPI.COMM_WORLD
            comm = pkl5.Intracomm(MPI.COMM_WORLD)


        nb1, nb2, nb3 = nb
        nbtot = np.prod(nb)

        b1, b2, b3 = np.meshgrid(range(nb1), range(nb2), range(nb3))
        nijk = [(ni, nj, nk) for ni, nj, nk in zip(b1.reshape(-1), b2.reshape(-1), b3.reshape(-1))]
        
        blist = range(comm.Get_rank(), nbtot, comm.Get_size())

        for r in blist:
            hroot, ext = os.path.splitext(self.header.head)
            bheadfile = hroot + bext + f"_{r}" + ext
            block = self.getBlock(bheadfile=bheadfile, nijk=nijk[r], nb=nb, verbose=verbose) 

            block.export()

        return 0


    def getBlock(self, nijk, nb, bheadfile, r=None, verbose=False):
        """
        Return a block partition of the original SEP model
        
        Parameters
        ----------
            nijk : 3d array-like
                Blocks number ID for each dimension
            nb : 3d array-like
                Total number of blocks for each dimension
            bheadfile : str
                Filename of the block
            r : int or str
                Block linear numbering (out of the total number of blocks)
            verbose : bool
                Print block information or not
        
        Returns
        --------
            SEPBlock
        """
        if not bheadfile:
            bheadfile = self.header.head
        block = SEPBlock(sepmodel=self, bfile=bheadfile, nijk=nijk, nb=nb)

        if verbose:
            if r:
                print(f"Block {r}")
            print("nijk:", nijk)
            print("Bounds:", block.header.getBounds())
            print("Number of elements:", block.header.getNumberOfElements())
            print("Index min:", block.imin)
            print("Index max:", block.imax)

        return block


    def getGlobalOrigin(self):
        """
        Return the global origin position of the model
            
        Returns
        --------
            array-like
        """
        return self.header.getOrigin()
   
 
    def getGlobalNumberOfElements(self):
        """
        Return the global number of elements for each dimension
            
        Returns
        --------
            array-like
        """
        return self.getNumberOfElements()


    def getNumberOfElements(self):
        """
        Return the number of elements for each dimension

        Returns
        -------
            array-like
        """
        return self.header.getNumberOfElements()


    def getStepSizes(self):
        """
        Return the step sizes for each dimension

        Returns
        -------
            array-like
        """
        return self.header.getStepSizes()


    def getBounds(self):
        """
        Return the min and max bounds of the model

        Returns
        -------
            array-like, array-like
        """
        return self.header.getBounds()


    def export(self, filename=None, directory=None):
        """
        Write header and binary in files
        
        Parameters
        ----------  
            filename : str
                New filename for the export
        """
        if filename is not None:
            outModel = self.copy(header=filename)
            header, binary = outModel.header, outModel.bin
        else:
            header, binary = self.header, self.bin

        header.write(directory=directory)
        binary.write(directory=directory)


    def copy(self, header=None):
        """
        Copy the SEP model

        Parameters
        ----------
            header : str
                New header filename

        Returns
        -------
            SEPModel
        """
        copyHead = self.header.copy(header)
        dictCopyHead = copyHead.convertToSEPDict()
        model = self.bin.copyRawData() 
        copyModel = SEPModel(dictCopyHead, model)

        return copyModel
    

    def getSlice(self, axis="y", index=None):
        """
        Return a slice of the 3d data

        Parameters
        -----------
            axis : str
                Slice axis \
                Default is `y`
            index : int
                Slice index \
                If None, default index is middle of axis

        Returns
        --------
            array-like
                2d slice of the original data
        """
        assert axis.lower() in ("x", "y", "z"), 'Unknown axis'
        data = self.getModel(datatype="points")
 
        if axis == "x":
            if index is None:
                index = int(data.shape[0]/2)
            return data[index,:,:]
        elif axis == "y":
            if index is None:
                index = int(data.shape[1]/2)
            return data[:,index,:]
        else:
            if index is None:
                index = int(data.shape[2]/2)
            return data[:,:,index]

    
    def display2d(self, axis="y", index=0, vmin=None, vmax=None, normFactor=False, xlabel="Receivers", ylabel="Time [s]"):
        """
        Generate a 2d graph for a given slice

        Parameters
        -----------
            axis : str, optional
                Slice axis \
                Default is `y`
            index : int, optional
                Slice index \
                Default is 0
            vmin : float, optional
                Minimal bound
                Default is None
            vmax : float, optional
                Maximal bound
                Default is None
            normFactor : float, optional
                Normalization factor
                Default is None
            xlabel : str
                Graph x label
            ylabel : str
                Graph y label
        """
        import matplotlib.pyplot as plt

        assert axis.lower() in ("x", "y", "z"), "Unknown axis"

        data = self.getSlice(axis=axis, index=index)

        if normFactor is not None:
            if vmin is None:   
                vmin = self.getMinValue()
            if vmax is None:
                vmax = self.getMaxValue()
            vmin /= normFactor
            vmax /= normFactor

        d = self.getStepSizes()
        n = list(self.getNumberOfElements())
        xy = [0,1,2]
        xy.remove(["x", "y", "z"].index(axis))
        n.remove(["x", "y", "z"].index(axis))

        x = np.arange(n[0])*d[xy[0]]
        y = np.arange(n[1])*d[xy[1]]

        # Graph
        plt.figure()

        Y, X = np.meshgrid(y,x)
        plt.pcolormesh(Y, X, np.transpose(data),
                       cmap="gray",
                       vmin=vmin, vmax=vmax)
        
        plt.ylim(x.max(), 0)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)


    def getMinValue(self):
        """
        Return the minimal value of the model

        Returns
        --------
            minValue : float
                Model minimal value
        """
        data = self.getModel(datatype="points")

        minValue = 0.
        for i in range(np.shape(data)[0]):
            if minValue > data[i,:,:].min():
                minValue = data[i,:,:].min()

        return minValue


    def getMaxValue(self):
        """
        Return the maximal value of the model

        Returns
        --------
            maxValue : float
                Model maximal value
        """
        data = self.getModel(datatype="points")

        maxValue = 0.
        for i in range(np.shape(data)[0]):
            if maxValue < data[i,:,:].max():
                maxValue = data[i,:,:].max()

        return maxValue


    def getMinAndMaxValues(self):
        """
        Return the minimal and maximal values of the model

        Returns
        --------
            minValue : float
                Model minimal value
            maxValue : float
                Model maximal value
        """
        data = self.getModel(datatype="points")

        minValue = 0.
        maxValue = 0.
        for i in range(np.shape(data)[0]):
            if minValue > data[i,:,:].min():
                minValue = data[i,:,:].min()
            if maxValue < data[i,:,:].max():
                maxValue = data[i,:,:].max()

        return minValue, maxValue


    def __repr__(self):
        rp = f"SEP Model. {self.getHeaderInfos()}" 
        rp += f" - {self.getBinInfos()}" 
        return rp
   


class SEPHeader:
    """
    Class defining a SEP Header file

    Attributes
    -----------
        head : str
            Header filename 
        bin : str
            Binary filename 
        n1, n2, n3 : int
            Number of elements in each dimension
        o1, o2, o3 : float
            Position of the origin
        d1, d2, d3 : float
            Step size for each dimension
        label1, label2, label3 : str
            Label for each dimension
        data_format : str
            Format type of the binary data
        esize : int
            Size of the elements in octet
        order : int
            order
    """
    def __init__(self, header):
        """
        Parameters
        ----------
            header : str or dict
                SEP header. Header filename if str, \
                  dict containing all the SEP header informations if dict,
        """
        if isinstance(header, str):
            self.head = header
            dictSEP = self.read()
            if "head" in dictSEP and self.head != dictSEP["head"]:
                dictSEP.update({"head" : header})
        elif isinstance(header, dict):
            dictSEP = self.convertToSEPDict(header.copy())
        else:
            print("Empty model")
            dictSEP = {}
        
        self._setAttributes(dictSEP) 


    def read(self):
        """
        Read a SEP Header

        Returns
        -------
            dict
        """
        try:
            with open(self.head, 'r') as f:
                headerStr = f.read()
        except:
            print(f"File {self.head} cannot be opened")
            sys.exit(1)        
        
        return self.parseStringToSEPDict(headerStr)        


    def _setAttributes(self, dictSEP):
        """Set class attributes from key-value pairs of a SEP dict
        """
        assert isinstance(dictSEP, dict)
        
        for k, v in dictSEP.items():
            if k == "in":
                setattr(self, "bin", v) #"in" attr raises Error in Python
            else:
                setattr(self, k, v) 

        self.correctBinPath() 

 
    def correctBinPath(self):
        """Correct the binfile path extracted from a .H file
        """
        if hasattr(self, "bin") and hasattr(self, "head"):
            b_path, b_file = os.path.split(self.bin)
            h_path, _ = os.path.split(self.head)
            
            if b_path != h_path:
                self.bin = os.path.join(h_path, b_file)

 
    def write(self, filename=None, directory=None):
        """
        Export the header

        Parameters
        ----------
            filename : str
                Filename in which to write the header. Default is self.head
        """
        if filename is None:
            if hasattr(self, "head"):
                _, filename = os.path.split(self.head)
                hcopy = self 
            else:
                raise FileNotFoundError("No header file set for the export")
        else:
            _, filename = os.path.split(filename)
            hcopy = self.copy(filename)
         
        headerStr = hcopy.getHeaderAsStr()

        if directory:
            filename = os.path.join(directory, filename)
        
        with open(filename, "w") as f:
            f.write(headerStr)
   
 
    def getHeaderAsStr(self):
        """
        Return the object as a SEP Header unique string
        
        Returns
        -------
            str
        """
        headerStr = ""
        for k, v in vars(self).items():
                if v is not None:
                    if k == "bin":
                        headerStr += f"in={v}\n"
                    else:
                        headerStr += f"{k}={v}\n"

        return headerStr


    def parseStringToSEPDict(self, headerStr):
        """
        Returns a dict containing the parsed options from a SEP header string

        Parameters
        ----------
            headerStr : str
                string read from a SEP header file
    
        Returns
        -------
            dict
        """
        if isinstance(headerStr, str):
            headerList = []
            for fl in headerStr.split('\n'):
                l = fl.split("#")[0]
                if l:
                    # add "--" to facilitate parsing that follows
                    l = "--" + l 
                    headerList += l.split("=")

            return self.parseListToSEPDict(headerList)


    def parseListToSEPDict(self, headerList):
        """
        Returns a dict from parsed SEP header list

        Parameters
        ----------
            headerList : list
                List of SEP options
            
        Returns
        -------
            dict
        """
        if isinstance(headerList, list):
            args, _ = self.SEPParser(headerList)

            return vars(args)


    def SEPParser(self, argsList=None):
        """
        SEP Header parser
        
        Parameters
        ----------
            argsList : list
               List of SEP options

        Returns
        -------
            Namespace, extra_args
        """
        parser = argparse.ArgumentParser("Parser for a SEP Header")

        n_parser = parser.add_argument_group("Number of elements", description="Number of elements for each dimension")
        n_parser.add_argument("--n1", type=int, default=None,
                            help="Number of elements for dimension 1")
        n_parser.add_argument("--n2", type=int, default=None,
                            help="Number of elements for dimension 2")
        n_parser.add_argument("--n3", type=int, default=None,
                            help="Number of elements for dimension 3")

        o_parser = parser.add_argument_group("Origin", description="Origin coordinates")
        o_parser.add_argument("--o1", type=float, default=None,
                            help="Coordinate 1 of the origin")
        o_parser.add_argument("--o2", type=float, default=None,
                            help="Coordinate 2 of the origin")
        o_parser.add_argument("--o3", type=float, default=None,
                            help="Coordinate 4 of the origin")

        d_parser = parser.add_argument_group("Step", description="Step size for each dimension")
        d_parser.add_argument("--d1", type=float, default=None,
                            help="Step size for dimension 1")
        d_parser.add_argument("--d2", type=float, default=None,
                            help="Step size for dimension 2")
        d_parser.add_argument("--d3", type=float, default=None,
                            help="Step size for dimension 3")

        f_parser = parser.add_argument_group("Files", description="Header and binary files")
        f_parser.add_argument("--in", "--bin", type=str, default=None,
                            help="Binary file")
        f_parser.add_argument("--head", type=str, default=None,
                            help="Header file")

        l_parser = parser.add_argument_group("Labels", description="Labels of each dimension")
        l_parser.add_argument("--label1", type=str, default=None,
                            help="Label for dimension 1")
        l_parser.add_argument("--label2", type=str, default=None,
                            help="Label for dimension 2")
        l_parser.add_argument("--label3", type=str, default=None,
                            help="Label for dimension 3")

        i_parser = parser.add_argument_group("Data information", description="Data format information")
        i_parser.add_argument("--data_format", type=str, #default="native_float",
                            help="Format type of binary data (default is little_endian)")
        i_parser.add_argument("--esize", type=int, default=4,
                            help="Size of the elements in octet (default is 4 = int/float)")
        i_parser.add_argument("--order", type=int, default=None,
                            help="Order")
        i_parser.add_argument("--data_type", type=str, default=None,
                            help="Type of data : cells or points")

        if argsList:
            return parser.parse_known_args(argsList)
        else:
            parser.parse_args(["--help"])
    

    def convertToSEPDict(self, genericDict=None, cleanNone=False):
        """ 
        Returns a SEP Header under dictionary format

        Parameters
        ----------
            genericDict : dict
                Dictionary to be converted to SEP Header dict. Default is self
            cleanNone : bool
                Remove None entries from dict or not. Default is False
        
        Returns
        -------
            dict
        """
        if genericDict is not None:
            assert isinstance(genericDict, dict)
            dictAsStr = ""
            for k, v in genericDict.items():
                if v is not None:
                    if k == "bin":
                        dictAsStr += f"in={v}\n"
                    else:
                        dictAsStr += f"{k}={v}\n"
        else:
            dictAsStr = self.getHeaderAsStr()    
            
        dictSEP = self.parseStringToSEPDict(dictAsStr)
        dictSEPOut = dictSEP.copy()
        if cleanNone:
            for k, v in dictSEP.items():
                if v is None:
                    dictSEPOut.pop(k)

        return dictSEPOut


    def copy(self, headfile=None):
        """
        Copy this SEPHeader
        
        Parameters
        ----------
            headfile : str (optional)
                New filename of the header. If none, the "head" entry is unaltered

        Returns
        -------
            SEPHeader
        """
        
        copyHead = self.convertToSEPDict()
        if headfile != None:
            copyHead.update({'head': f"{headfile}"})
            copyHead.update({'in': f"{headfile}@"})
        
        return SEPHeader(copyHead)


    def getNumberOfElements(self):
        """
        Return the number of elements for each dimension

        Returns
        -------
            tuple of int
        """
        return (self.n1, self.n2, self.n3)


    def setNumberOfElements(self, n):
        """
        Update the number of elements for each dimension

        Parameters
        ----------
            3d array or tuple : n
                New number of elements for each dimension
        """
        assert len(n)==3
        self.n1, self.n2, self.n3 = n


    def getStepSizes(self):
        """
        Return the step sizes for each dimension
        
        Returns
        ----------
            tuple of float
        """
        return (self.d1, self.d2, self.d3)


    def getOrigin(self):
        """
        Return the origin position
        
        Returns
        --------
            tuple of float
        """
        return (self.o1, self.o2, self.o3)


    def setOrigin(self, origin):
        """
        Set the origin of the model

        Parameters
        ----------
            3d array or tuple : origin
                New origin coordinates
        """
        self.o1, self.o2, self.o3 = origin


    def getBounds(self):
        """
        Return the bounds of the model
        
        Returns
        -------
            tuple of float, tuple fo float
        """
        bmins = self.getOrigin()
        n = self.getNumberOfElements()
        d = self.getStepSizes()
        if not(None in bmins) and not(None in n) and not(None in d):    
            if self.data_type == "cells":
                bmax = tuple(np.array(bmins)+(np.array(n)-1)*np.array(d))
            else:
                bmax = tuple(np.array(bmins)+np.array(n)*np.array(d))
        else:
            bmax = (None, None, None)

        return bmins, bmax


    def getLabels(self):
        """
        Return the label for each dimension
        
        Returns
        -------
            tuple of str
        """
        return (self.label1, self.label2, self.label3)


    def setLabels(self, labels):
        """
        Set new labels for each dimension

        Parameters
        ----------
            labels : 3d str
                New labels
        """
        assert len(labels)==3
        self.label1, self.label2, self.label3 = labels



class SEPBin:
    """
    Class defining a SEP binary file

    Attributes
    ----------
        dataFormat : str
            Format type of the binary data
        bin : str
            SEP binary file
        head : str
            Associated SEP header file
        n : tuple of int
            Number of elements in each dimension
        dataType : str
            Type of data : cells or points
        data : numpy array
            Model values
    """
    def __init__(self, binfile=None, header=None, data=None, **kwargs):
        """
        Parameters
        ----------
            binfile : str
                Binary filename
            header : SEPHeader
                Header associated to this binary SEP file
            data : numpy array
                Model values
            kwargs :
                datatype : "cells" or "point". Default is None
        """
        if header:
            dictHeader = header.convertToSEPDict(cleanNone=False)
            self.n = (dictHeader["n1"], dictHeader["n2"], dictHeader["n3"])
            self.dataFormat = dictHeader["data_format"]
            if self.dataFormat is not None:
                self.dataFormat = self.dataFormat.replace('"','')
            self.bin = dictHeader["in"]
            self.head = dictHeader["head"]
            self.datatype = dictHeader["data_type"]
            if self.datatype is not None:
                self.datatype = self.datatype.replace('"','')
 
        elif not header and binfile:
            self.bin = str(binfile)
            self.head, _ = binfile.split("@")
            self.dataFormat = None 
            self.datatype = None
        else:
            raise ValueError("You need to specify a SEP binary file parameter or a SEPHeader in order to initialize a SEPBin object")

        if data is not None:
            self.data = data
            if not hasattr(self, "n"): 
                self.n = np.shape(data)
            if "datatype" in kwargs:
                self.datatype = kwargs["datatype"]
        else:
            self.data = self.read()  
    

    def getModel(self, datatype=None):
        """
        Get the cell or point data from the binary file

        Parameters
        ----------
            datatype : str
                Type of requested data: 'cells' or 'points'

        Returns
        -------
            numpy array
        """
        if self.data is None:
            data = self.read(transp=True)
        else:
            data = self.reshapeData(self.data)
            

        if datatype:
            datatype = datatype.lower()
        elif not datatype and self.datatype:
            datatype = self.datatype
        else:
            datatype = "cells"

        if datatype not in ["cells", "points"]:
            raise ValueError(f"Element datatype can only be 'cells' or 'points' or raw, not {datatype}")    
        elif datatype == "cells" and self.datatype != "cells":
            model = data[:-1,:-1,:-1]
        else:
            model = data
 
        return model


    def read(self, transp=False, **kwargs):
        """
        Read data from the binary file. If header provided, can reshape the data
    
        Parameters
        ----------
            transp : bool
                Whether or not to reshape the data. Default is False is no header provided, True otherwise.
            kwargs :
                data_format : data format (little/big endian)
        """
        # Set data format
        if self.dataFormat:
            dform = self.dataFormat
        elif not self.dataFormat and "data_format" in kwargs:
            dform = kwargs["data_format"]
            self.dataFormat = dform
        else:
            dform = None

        if dform == "native_float" or dform == "little" or dform is None:
            fdata = "<f4" #little endian
        else:
            fdata = ">f4" #big endian

        # Reading
        try:
            data = np.fromfile(self.bin, dtype=fdata)
        except:
            print("Unable to read data:")
            print(f"data format: {fdata} from binary file '{self.bin}'.")
            sys.exit(1)
 
        # Reshape if requested
        if transp:
            data = self.reshapeData(data)
        return data


    def reshapeData(self, data, **kwargs):
        """
        Reshape data with correct number of elements in each dimension

        Parameters
        ----------
            data : numpy array
                Model data
            kwargs :
                Should contain n1, n2 and n3 if self.n is all None

        Returns
        -------
            numpy array
        """
        if not(None in self.n):
            n = tuple(reversed(self.n))

        elif "n1" in kwargs and "n2" in kwargs and "n3" in kwargs:
            n1 = kwargs["n1"]
            n2 = kwargs["n2"]
            n3 = kwargs["n3"]
            assert(isinstance(n1, int) and isinstance(n2, int) and isinstance(n3, int))
            n = (n3, n2, n1)

        else:
            print("Number of elements in each dimension (n1, n2, n3) required to reshape data from the binary file")
            print("data unaltered")
            return data

        data = data.reshape(n)
        return data


    def write(self, directory=None):
        """Export the data to a file
        """
        model = self.data.astype(np.float32)
        if directory:
            filename = os.path.join(directory, self.bin)
        model.tofile(filename, sep="")
        

    def copy(self, binfile=None, header=None):
        """Copy all binary SEPBin object
        """
        if binfile is None:
            binfile = self.bin

        if header:
            if isinstance(header, str):
                header = SEPHeader(header)
            elif isinstance(header, SEPHeader):
                headerfile, _ = binfile.split("@")
                header = header.copy(headerfile)

        copyBin = SEPBin(binfile=binfile, header=header, data=self.data.copy())
        return copyBin


    def copyRawData(self):
        """
        Return a copy of the data array

        Returns
        -------
            numpy array
        """
        return self.data.copy()

 
    def __repr__(self):
        rp = f"SEP Binary file: {self.bin} associated to header file: {self.head}"
        if self.data is not None:
            rp += f" - Length of data: {np.shape(self.data)}"
        return rp



class SEPBlock(SEPModel):
    """
    SEP block
    Inheritance from SEPModel
        
    Attributes
    ----------
        gModel : str
            Global model filename
        header : SEPHeader
            Contains SEP header information
        bin : SEPBin
            Contains SEP binary information
        nijk : array-like of int
            Block number identification
        nblocks : array-like of int
            Total number of blocks in each dimension
        imin : tuple of int
            Index min of the block in the whole dataset
        imax : tuple of int
            Index max of the block in the whole dataset
        n : tuple of int
            Number of elements in the block for each dimension
        bmin : tuple of float
            Origin of the block
        gln : tuple of int
            Global number of elements for each dimension
        glmin : tuple of float
            Global origin of the model
    """
    def __init__(self, sepmodel, bfile, nijk, nb):
        """
        Parameters
        ----------
            sepmodel : SEPModel
                Global model
            bfile : str
                Block filename
            nijk : array-like of int
                Block number identification
            nb : array-like of int
                Total number of blocks in each dimension
        """
        self.gModel = sepmodel.header.head
        name = sepmodel.name
        self.nijk = nijk
        self.nblocks = nb

        # First, copy the original SEPModel with new filename
        hblock = sepmodel.header.copy(bfile)
        dictHblock = hblock.convertToSEPDict()
        data = sepmodel.bin.copyRawData()

        SEPModel.__init__(self, header=dictHblock, data=data, name=name)
        self.glmin, _ = self.getBounds()
        self.gln = self.getNumberOfElements()

        # Reshape the data
        self.bin.data = self.bin.reshapeData(data)

        # Now we need block index min et max
        # As well as the block number of elements      
        self.__setCroppingIndex() 

        # Define the new origin
        self.__setOrigin()
    
        #Finally, crop the data !
        self.__setData()

        #And update the properties that have been changed
        self.__update()             


    def __setNumberOfElementsAndIndexMax(self):
        """Compute and set the index max of the block and the number of elements for each dimension
        """
        n = self.header.getNumberOfElements() #original SEP
        bn = [None, None, None] # number elts of this block
        imax = [None, None, None]

        for dim in range(3):
            if self.nijk[dim] == self.nblocks[dim] - 1:
                imax[dim] = n[dim]
                bn[dim] = imax[dim] - self.imin[dim]
            else:
                bn[dim] = n[dim] // self.nblocks[dim] + 1 
                imax[dim] = self.imin[dim] + bn[dim]

        self.imax = tuple(imax)
        self.n = tuple(bn) 

 
    def __setOrigin(self):
        """Compute and set the origin of the block
        """
        sepmin, _ = self.header.getBounds()
        sepd = self.header.getStepSizes()
        bmin = np.array(sepmin) + np.array(self.imin) * np.array(sepd)

        self.bmin = tuple(bmin)


    def __setCroppingIndex(self):
        """Set the min, max indices of the block as well as the resulting number of elements
        """
        self.__setIndexMin()
        self.__setNumberOfElementsAndIndexMax()


    def __setData(self):
        """Resize and set block data
        """        
        min1, min2, min3 = self.imin
        max1, max2, max3 = self.imax

        self.data = self.bin.data[min3:max3, min2:max2, min1:max1]
        self.data = self.data.reshape(np.prod(self.n)) 


    def __setIndexMin(self):
        """Compute and set the min indices of the block
        """
        imin = np.array(self.nijk)*(np.array(self.header.getNumberOfElements())//np.array(self.nblocks))
        self.imin = tuple(imin)


    def __update(self):
        """Update header and binary properties after initialization
        """
        self.__updateHeader()
        self.__updateBin()


    def __updateHeader(self):
        """Update header properties changed due to block initialization
        """
        self.header.setOrigin(self.bmin)
        self.header.setNumberOfElements(self.n)


    def __updateBin(self):
        """Update binary properties modified due to block initialiation
        """
        self.bin.n = self.n
        self.bin.data = self.data
        
    
    def getGlobalOrigin(self):
        """
        Return the global origin of the model
        
        Returns
        -------
            array-like
        """
        return self.glmin


    def getGlobalNumberOfElements(self):
        """
        Return the global number of elements of the model
        
        Returns
        -------
            array-like
        """
        return self.gln
