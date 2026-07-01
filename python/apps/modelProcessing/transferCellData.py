import sys
import numpy as np
import argparse
import vtk
from utilities.mesh import VTKMesh

def parse_args():
    """Get arguments

    Returns:
        argument '--infile': Mesh containing the input cell data, in vtu format
        argument '--outfile': Mesh where the cell data will be transferred, in vtu format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help="Mesh containing the input cell data, in vtu format", required=True)
    parser.add_argument('--outfile', help="Mesh where the cell data will be transferred, in vtu format", required=True)

    args,_ = parser.parse_known_args()
    return args


def main():
    global point2Index, globalId, tet_mesh, hex_mesh_points

    args = parse_args()
    in_mesh_file = args.infile
    out_mesh_file = args.outfile

    # read input mesh
 
    print(f'Reading file {in_mesh_file}...')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(in_mesh_file)
    reader.Update()
    in_mesh = reader.GetOutput()

    print(f'Read input mesh with {in_mesh.GetNumberOfPoints()} vertices and {in_mesh.GetNumberOfCells()} cells.')
 
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
    centers = cell_centers.GetOutput(0).GetPoints()
    # get data at cell centers
    print("Interpolating data...")
    dest = vtk.vtkPointSet()
    dest.SetPoints(centers)
    resample = vtk.vtkResampleWithDataSet()
    resample.SetSourceData(in_mesh)
    resample.SetInputData(dest)
    resample.Update()
    pointdata = resample.GetOutput().GetPointData()
    print("Adding arrays to output data")
    for i in range(pointdata.GetNumberOfArrays()):
        array = pointdata.GetArray(i)
        print(f"Adding array {array.GetName()} to output data")
        out_mesh.GetCellData().AddArray(array)
    
    print(f'Writing mesh into file {out_mesh_file}...')
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out_mesh_file)
    writer.SetInputData(out_mesh)
    writer.Write()

    print("Done.")


if __name__ == "__main__":
    main()
