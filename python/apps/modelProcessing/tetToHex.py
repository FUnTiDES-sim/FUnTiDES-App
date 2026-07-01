import sys
import numpy as np
import argparse
import vtk
from statistics import mean

def parse_args():
    """Get arguments

    Returns:
        argument '--infile': Input tetrahedral mesh, in vtu format
        argument '--outfile': Output hexahedral mesh, (vtu format)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', help="Input tetrahedral mesh, in vtu format", required=True)
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
    tet_mesh_file = args.infile
    hex_mesh_file = args.outfile

    # read tet mesh
    print("                                 .+------+")
    print("       ^                       .\' |    .\'|")
    print("      /|\               ┌,    +---+--+\'  |")
    print("     / | \    ┌─────────┘ \   |   |  |   |")
    print("    /  |  \   └─────────┐ /   |  ,+--+---+")
    print("   /___|___\            └\'    | .\'   | .\' ")
    print("   \'-. | .-\'                  +------+\'   ")
    print("       ¯                                     ")
 
    print(f'Reading file {tet_mesh_file}...')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(tet_mesh_file)
    reader.Update()
    tet_mesh = reader.GetOutput()
    tet_mesh_points = tet_mesh.GetPoints()
    cells = tet_mesh.GetCells()

    print(f'Read tetrahedral mesh with {tet_mesh.GetNumberOfPoints()} vertices and {tet_mesh.GetNumberOfCells()} tetrahedra.')
 
    # create output mesh
    hex_mesh = vtk.vtkUnstructuredGrid()
    hex_mesh_points=vtk.vtkPoints()
    hex_mesh_cells=vtk.vtkCellArray()

    # maps for existing and new vertices
    point2Index = {}
    
    # start splitting cells
    print('Splitting cells...')
    globalId = 0
    for t in range(cells.GetNumberOfCells()):
        if t % 10000 == 0:
            print(f'Cell {t}/{cells.GetNumberOfCells()}',end='\r')
        cell = tet_mesh.GetCell(t)
        cellPointIds = cell.GetPointIds()
        v = [cellPointIds.GetId(i) for i in range(cellPointIds.GetNumberOfIds())]
        # build the four hex
        for shift in range(4):
            if shift % 2 == 0: 
                hex_mesh_cells.InsertNextCell(8,
                                   (getId((-1, -1, -1, v[0])),
                                    getId((-1, -1, v[0], v[1])),
                                    getId((-1, v[0], v[1], v[2])),
                                    getId((-1, -1, v[0], v[2])),
                                    getId((-1, -1, v[0], v[3])),
                                    getId((-1, v[0], v[3], v[1])),
                                    getId((v[0], v[1], v[2], v[3])),
                                    getId((-1, v[0], v[2], v[3]))))
            else:
                hex_mesh_cells.InsertNextCell(8,
                                   (getId((-1, -1, v[0], v[3])),
                                    getId((-1, v[0], v[3], v[1])),
                                    getId((v[0], v[1], v[2], v[3])),
                                    getId((-1, v[0], v[2], v[3])),
                                    getId((-1, -1, -1, v[0])),
                                    getId((-1, -1, v[0], v[1])),
                                    getId((-1, v[0], v[1], v[2])),
                                    getId((-1, -1, v[0], v[2]))))
            v = v[1:] + v[:1]
    print(f'Created cells for hehxahedral mesh. A total of {hex_mesh_points.GetNumberOfPoints()} vertices and {hex_mesh_cells.GetNumberOfCells()} hexahera have been created.')
    
    # write mesh
    print(f'Writing hexahedral mesh into file {hex_mesh_file}...')
    hex_mesh.SetPoints(hex_mesh_points)
    hex_mesh.SetCells(vtk.VTK_HEXAHEDRON, hex_mesh_cells)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(hex_mesh_file)
    writer.SetInputData(hex_mesh)
    writer.Write()
    print("Done.")


if __name__ == "__main__":
    main()
