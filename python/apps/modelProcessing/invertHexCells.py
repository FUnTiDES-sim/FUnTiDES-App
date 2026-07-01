import sys
import numpy as np
import argparse
import vtk
from statistics import mean

def parse_args():
    """Get arguments

    Returns:
        argument '--infile': Input hexahedral mesh, in vtu format
        argument '--outfile': Output hexahedral mesh, (vtu format)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help="Input hexahedral mesh, in vtu format", required=True)
    parser.add_argument('--outfile', help="Output hexahedral mesh (vtu format)", required=True)

    args,_ = parser.parse_known_args()
    return args

def getId(pt):
    global point2Index, globalId, points, tet_mesh, hex_mesh_points
    pt = tuple(sorted(pt))
    if pt in point2Index:
        return point2Index[pt]
    else:
        point2Index[pt] = globalId
        originalPoints =[tet_mesh.GetPoint(p) for p in pt if p >= 0]
        newPointCoords = (mean(originalPoints[i][0] for i in range(len(originalPoints))),
                          mean(originalPoints[i][1] for i in range(len(originalPoints))),
                          mean(originalPoints[i][2] for i in range(len(originalPoints))))
        hex_mesh_points.InsertNextPoint(newPointCoords)
        globalId = globalId + 1
        return globalId - 1 

def main():
    global point2Index, globalId, tet_mesh, hex_mesh_points

    args = parse_args()
    in_mesh_file = args.infile
    out_mesh_file = args.outfile

    # read tet mesh
 
    print(f'Reading file {in_mesh_file}...')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(in_mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    cells = mesh.GetCells()

    print(f'Read hexahedral mesh with {mesh.GetNumberOfCells()} hexahedra.')
 
    # create output mesh
    print(f'Creating output mesh...')
    out_mesh = vtk.vtkUnstructuredGrid()
    out_mesh_points=vtk.vtkPoints()
    out_mesh_cells=vtk.vtkCellArray()
    for p in range(mesh.GetNumberOfPoints()):
        out_mesh_points.InsertNextPoint(mesh.GetPoint(p))   
    out_mesh.SetPoints(out_mesh_points) 
    for i in range(mesh.GetPointData().GetNumberOfArrays()):
        out_mesh.GetPointData().AddArray(mesh.GetPointData().GetArray(i)) 
    for i in range(mesh.GetCellData().GetNumberOfArrays()):
        out_mesh.GetCellData().AddArray(mesh.GetCellData().GetArray(i)) 
  
    # start inverting cells
    print('Inverting hexahedra...')
    for t in range(cells.GetNumberOfCells()):
        if t % 100000 == 0:
            print(f'Cell {t}/{cells.GetNumberOfCells()}',end='\r')
        cell = mesh.GetCell(t)
        cellPointIds = cell.GetPointIds()
        v = [cellPointIds.GetId(i) for i in range(cellPointIds.GetNumberOfIds())]
        out_mesh_cells.InsertNextCell(8, (v[4], v[5], v[6], v[7], v[0], v[1], v[2], v[3]))
    out_mesh.SetCells(vtk.VTK_HEXAHEDRON, out_mesh_cells)

    # write mesh
    print(f'Writing hexahedral mesh into file {out_mesh_file}...')
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out_mesh_file)
    writer.SetInputData(out_mesh)
    writer.Write()
    print("Done.")


if __name__ == "__main__":
    main()
