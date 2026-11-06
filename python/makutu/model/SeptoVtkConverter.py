import os
import sys
import numpy as np
import argparse

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI
from mpi4py.util import pkl5

from utilities.model.utils.SepModel import SEPModel, SEPHeader
from utilities.model.utils.VtkModel import VTSModel, VTUModel, PVTKModel

if not MPI.Is_initialized():
    MPI.Init()

class SeptoVtkConverter:
    """
    Class handling the conversion of models (velocity, density, ..)

    Attributes
    -----------
        infile : str
            Input file.
            For multi-dataset, it should be a file containing all the names and filenames.
            Otherwise, it can be directly the input model file 
        outformat : str
            Output format
        outdir : str
            Output directory. If none, set as the directory of infile
        nblocks : int
            Number of blocks
            Required if conversion from VTS -> PVTS
        nb1, nb2, nb3 : int, int, int
            Number of blocks for each dimension
            Required if SEP -> PVTS
        propname : str
            Property name if only one property is required
        gids : bool
            Compute or not the global Ids
        verbose : bool
            Print information on conversion or not
        dataset : dict
            Dictionary containing all dataset names and corresponding files
    """
    def __init__(self):
        args = vars(self.optionParser())
        
        for k,v in args.items():
            setattr(self, k, v)

        # Extract dataset information
        if self.mf:
            with open(self.infile, "r") as f:
                self.dataset = dict([l.split() for l in f])
            if len(self.dataset.keys()) > 1:
                ext = self.checkSameHeader()
        else:
            self.dataset = {self.propname: self.infile}
            _, ext = os.path.splitext(self.infile)

        self.intype = ext[1:]
        assert self.intype in ["vts", "vtu", "pvts", "pvtu", "H"]

        # Set output information
        if not self.outroot:
            if not self.mf:
                self.outroot, _ = os.path.split(self.infile)[-1].split(".")
            else:
                raise ValueError("'outrootname' argument required for multi dataset conversion")    

        if not self.outdir:
            self.outdir, _ = os.path.split(self.infile)
            if not self.outdir:
                self.outdir = "./"
        
        self.outroot = os.path.join(self.outdir, self.outroot)
        
        # Create the dir if it does not exist
        os.makedirs(self.outdir, exist_ok=True)

        # Partition information
        if self.nb1 and self.nb2 and self.nb3:
            self.nbtot = np.prod([self.nb1, self.nb2, self.nb3])        


    def checkSameHeader(self):
        """Assert that all models headers have the same number of elements, step sizes and bounds

        Returns
        --------
            str
                Extension of the dataset files
        """
        # First, check the extensions
        pkeys = list(self.dataset.keys())
        _, ext = os.path.splitext(self.dataset[pkeys[-1]])
        assert all(os.path.splitext(self.dataset[k])[-1] == ext for k in pkeys)

        # Then check headers
        if self.verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("Checking if all headers are consistent")

        if ext == ".H":
            head1 = SEPHeader(self.dataset[pkeys[-1]])
            n = head1.getNumberOfElements()
            d = head1.getStepSizes()
            bmin, bmax = head1.getBounds()

            for hname in pkeys[:-1]:
                head = SEPHeader(self.dataset[hname])
                assert head.getNumberOfElements() == n
                assert head.getStepSizes() == d
                assert head1.getBounds()[0] == bmin
        
        return ext


    def optionParser(self):
        """
        Model conversion parser.
        Print all the options with --help
        """
        p = argparse.ArgumentParser("Model Conversion Parser")

        f_parser = p.add_argument_group("Format options")
        f_parser.add_argument("--infile", dest="infile", type=str, required=True,
                        help="Input file. Name of the input file for the dataset OR file containing the headers and model names required")
        f_parser.add_argument("--out", "--outformat", choices=["vts", "vtu", "pvts", "pvtu", "H"], dest="outformat",
                        help="Output file format", required=True)
        f_parser.add_argument("--outdir", dest="outdir", type=str, default=None,
                        help="Output directory if different from input directory")
        f_parser.add_argument("--or", "--outrootname", type=str, dest="outroot", default=None,
                        help="Root for the output file. Required if no infile is given")
        f_parser.add_argument("--dtype", type=str,
                            choices=("points", "cells"),
                            default="cells",
                            help="Output datatype \
                                `point` for vtkPointData, `cell` for vtkCellData ")

        mpn_parser = f_parser.add_mutually_exclusive_group()
        mpn_parser.add_argument("--mf", dest="mf", action="store_true",
                        help="Flag indicating if the models should be extracted from several files")        
        mpn_parser.add_argument("--pn", "--propname", dest="propname", type=str, default=None, 
                        help="Name of the model property. Only if not multifiles")

        p_parser = p.add_argument_group("Partition options")
        p_parser.add_argument("--nb", "--nblocks", type=int, default=1,
                        help="Number of blocks")
        p_parser.add_argument("--nb1", dest="nb1", type=int, default=None, required=False,
                        help="Number of blocks for the dimension 1")
        p_parser.add_argument("--nb2", dest="nb2", type=int, default=None, required=False,
                        help="Number of blocks for the dimension 2")
        p_parser.add_argument("--nb3", dest="nb3", type=int, default=None, required=False,
                        help="Number of blocks for the dimension 3")

        o_parser = p.add_argument_group("Other options")
        o_parser.add_argument("--gids", "--globalIds", dest="gids", action="store_true",
                        help="Generate global IDs for the output files") 
        o_parser.add_argument("--v", "--verbose", dest="verbose",action="store_true", default=False,
                        help="Print conversion information")

        args, _ = p.parse_known_args()

        return args 


    def convert(self):
        """ Trigger the conversion requested
        """
        #comm = MPI.COMM_WORLD
        comm = pkl5.Intracomm(MPI.COMM_WORLD)

        if self.intype == self.outformat and self.nbtot == 1:
            print("Same input and output type")
            sys.exit(1)

        # Conversion from SEP format
        if self.intype == "H":
            # Define the input models
            inModelList = []
            for dname, dfile in self.dataset.items():
                inModelList += [SEPModel(dfile, name=dname)]
            
            if not inModelList:
                print("Problem reading the property file")
                sys.exit(1)
            
            # SEP to SEP
            if self.outformat == "H":
                nblocks = (self.nb1, self.nb2, self.nb3)

                if None not in nblocks:
                    for i, inModel in enumerate(inModelList):
                        if self.verbose:
                            print(f"Dataset {i}: '{inModel.name}'")
                        inModel.createAllBlocks(nblocks, verbose=self.verbose, comm=comm)


            # SEP to VTK
            elif self.outformat == "vts" or self.outformat == "vtu":
                outfile = self.outroot + '.' + self.outformat
                if comm.Get_rank() == 0:
                    print("Output file name :", outfile)

                if self.outformat == "vts":
                    outModel = VTSModel(outfile)
                elif self.outformat == "vtu":
                    outModel = VTUModel(outfile)

                self.SEPToVTK(inModelList, outModel)

            # SEP to PVTK
            elif self.outformat == "pvts" or self.outformat == "pvtu":
                outfile = self.outroot + "." + self.outformat
                if comm.Get_rank() == 0:
                    print("Output file name :", outfile)

                nblocks = (self.nb1, self.nb2, self.nb3)
                modelExtent = inModelList[0].getGlobalNumberOfElements()

                if None not in nblocks:
                    pmodel = PVTKModel(outfile)
                    pmodel.setNumberOfBlocks(nblocks)
                    pmodel.setWholeExtent(modelExtent) 

                    self.SEPtoPVTK(inModelList, pmodel, comm=comm)
                    

    def SEPToVTK(self, sepModelList, vtkModel, isBlock=False, **kwargs):
        """
        Convert .H model(s) to VTK structured (.vts) or unstructured grid format (.vtu)
        
        Parameters
        ----------
            sepModelList : list of SEPModel
                List of model in SEP format
            vtkModel : VTUModel or VTSModel
                Final format model
            isBlock : bool
                Whether or not the output vtkmodel is a partition of the intput model
                If True, keyword args should be provided
            **kwargs
                nblocks : tuple of int
                    Total number of blocks for each dimension
                nijk : tuple of int
                    Blocks number Ids for each dimension
                bheadfile : str
                    Filename of the block

        Returns
        --------
            VTKModel
                Output VTK model with correct parameters
        """
        for sepModel in sepModelList:
            if self.verbose:
                print(f"Dealing with '{sepModel.name}'")

            if isBlock:
                nblocks = kwargs["nblocks"]
                nijk = kwargs["nijk"]
                bheadfile = kwargs["bheadfile"]

                sepmodel = sepModel.getBlock(nijk=nijk, nb=nblocks, bheadfile=bheadfile, verbose=self.verbose)
            else:
                sepmodel = sepModel

            model = sepmodel.getModel(datatype=self.dtype)
            ncells = np.add(sepmodel.getNumberOfElements(), -1)

            if self.dtype == "cells":
                model = model.reshape(np.prod(ncells))
                vtkModel.addCellData(sepmodel.name, model)
            else:
                model = model.reshape(np.prod(sepmodel.getNumberOfElements()))
                vtkModel.addPointData(sepmodel.name, model)
            

        n = sepmodel.getNumberOfElements()
        d = sepmodel.getStepSizes()
        bmin, _ = sepmodel.getBounds()
    
        gln = sepmodel.getGlobalNumberOfElements()
        glmin = sepmodel.getGlobalOrigin()

        # Set VTK parameters
        vtkModel.setNumberOfElements(n)
        vtkModel.setGlobalNumberOfElements(gln)
        vtkModel.setOrigin(bmin)
        vtkModel.setStepSize(d)
        vtkModel.setGlobalOrigin(glmin)

        if vtkModel.vtktype == "vtu":
            # Define cell types
            ncells = np.prod(np.array(vtkModel.n)-1)
            vtkModel.setCellsType(ncells=ncells, ctype=12) #vtk.VtkHexahedron.tid
        
        # Write .vtk file
        vtkModel.export(gids=self.gids)
        return vtkModel


    def SEPtoPVTK(self, sepModelList, pvtkModel, comm=None):
        """
        Convert a SEP dataset to a PVTK format.

        Parameters
        ----------
            sepModelList : list
                List of SEPModels to convert
            pvtkModel : PVTKModel
                PVKT model for the output
        """
        rank = comm.Get_rank()

        vtkroot = pvtkModel.rootname
        nblocks = pvtkModel.nblocks
        nbtot = pvtkModel.nbltot

        # Create the nb1 x nb2 x nb3 blocks
        nijk = pvtkModel.getAllBlocksIndices()
        blist = range(rank, nbtot, comm.Get_size())

        bfiles = {}
        for r in blist:
            if self.verbose:
                print("Rank ", rank, r, nijk[r])
            if self.outformat == "pvts":
                bVtkModel = VTSModel(vtkroot+f"_{r}.vts")
            else:
                bVtkModel = VTUModel(vtkroot+f"_{r}.vtu")
            

            bVtkModel = self.SEPToVTK(sepModelList, bVtkModel, isBlock=True, nijk=nijk[r], nblocks=nblocks, bheadfile=None, verbose=self.verbose)
            
            pvtkModel.addBlockInfo(bVtkModel, bijk=nijk[r])
            bfiles = {**bfiles, f"bfile{nijk[r]}": bVtkModel.filename}

        bInfo = {f"cinfo{rank}": pvtkModel.getCellInfo(),
                     f"pinfo{rank}": pvtkModel.getPointInfo(),
                     **bfiles
            }
        #Add extent info if structured grid
        if self.outformat == "pvts":
            bInfo = {**bInfo, **pvtkModel.getBlocksStarts(), **pvtkModel.getBlocksEnds()}


        if rank != 0: 
            comm.send(bInfo, dest=0, tag=1)  
        else:
            for r in range(1, comm.Get_size()):
                rInfo = comm.recv(source=r, tag=1)
                bInfo = {**bInfo, **rInfo}
                
                pvtkModel.addCellInfo(bInfo[f"cinfo{r}"])
                pvtkModel.addPointInfo(bInfo[f"pinfo{r}"])
            
            for nb in range(len(nijk)):
                if nb not in blist:
                    bfile = {nb: bInfo[f"bfile{nijk[nb]}"]}

                    if self.outformat == "pvts":
                        start = bInfo[(*nijk[nb], 0)]
                        end = bInfo[(*nijk[nb], 1)]
                    else:
                        start = None
                        end = None

                    pvtkModel.addSourceInfo(bijk = nijk[nb], bfile=bfile, cinfo=None, pinfo=None, start=start, end=end)
            
            pvtkModel.export(gids=self.gids, writeBlocks=False)  
