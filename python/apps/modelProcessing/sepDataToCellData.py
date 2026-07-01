import sys
import numpy as np
import argparse
import vtk
from vtk.util import numpy_support as VN
from utilities.model.utils.SepModel import SEPModel, SEPBin, SEPHeader

def parse_args():
    """Get arguments

    Returns:
        argument '--infile': SEP file containing the input cell data
        argument '--outfile': Mesh where hthe cell data will be transferred, in vtu format
        argument '--propname': Name of the property in the vtu file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help="SEP file containing the input cell data", required=True)
    parser.add_argument('--outfile', help="Mesh where the cell data will be transferred, in vtu format", required=True)
    parser.add_argument('--propname', help="Name of the property in the vtu file", required=True)

    args,_ = parser.parse_known_args()
    return args


def main():
    global point2Index, globalId, tet_mesh, hex_mesh_points

    args = parse_args()
    in_sep_file = args.infile
    out_mesh_file = args.outfile
    prop_name = args.propname

    # read input SEP
 
    print(f'Reading file {in_sep_file}...')
    sepmodel = SEPModel(header=in_sep_file)
    head = sepmodel.header
    n = head.getNumberOfElements()
    d = head.getStepSizes()
    bmin, bmax = head.getBounds()
    o = head.getOrigin()
    model = sepmodel.getModel(datatype="points")


    print(f'Read input SEP, origin {o}, size {n} and step {d}')
 
    # read output mesh
    print(f'Reading file {out_mesh_file}...')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(out_mesh_file)
    reader.Update()
    out_mesh = reader.GetOutput()

    print(f'Read output mesh with {out_mesh.GetNumberOfPoints()} vertices and {out_mesh.GetNumberOfCells()} cells.')

    
    # retrieve cell centers
    print("Computing cell centers...")
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(out_mesh)
    cell_centers.Update()
    centers = cell_centers.GetOutput(0)
    print(f"Computed {centers.GetNumberOfPoints()} centers.")
    array_data = np.zeros(centers.GetNumberOfPoints())
    # get data at cell centers
    print("Interpolating data...")
    for p in range(centers.GetNumberOfPoints()):
        if p % 100000 == 0:
            print(f'Computing value for cell {p}/{centers.GetNumberOfPoints()}...',end='\r')
        pt = centers.GetPoint(p)
        i = int(np.floor((pt[0] - o[0] ) / d[0]))
        j = int(np.floor((pt[1] - o[1] ) / d[1]))
        k = int(np.floor((pt[2] - o[2] ) / d[2]))
        array_data[p] = model[k,j,i] 
    print("Adding array to output data")
    array_data_vtk = VN.numpy_to_vtk(num_array=array_data.ravel(), deep=True)
    array_data_vtk.SetName(prop_name)
    out_mesh.GetCellData().AddArray(array_data_vtk)
    
    print(f'Writing mesh into file {out_mesh_file}...')
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out_mesh_file)
    writer.SetInputData(out_mesh)
    writer.Write()

    print("Done.")


if __name__ == "__main__":
    main()
