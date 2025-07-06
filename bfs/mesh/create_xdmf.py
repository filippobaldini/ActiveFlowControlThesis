import meshio
from dolfin import *
import numpy as np


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

    # Optional: remove untagged
    # mask = cell_data != 18446744073709551615
    # cells = cells[mask]
    # cell_data = cell_data[mask]
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data.astype(np.int32)]},
    )
    return out_mesh


proc = MPI.comm_world.rank


if proc == 0:
    # Read in mesh
    msh = meshio.read("our_mesh.msh")

    # Create and save one file for the mesh, and one file for the facets
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("our_mesh.xdmf", triangle_mesh)
    meshio.write("our_mesh_facets.xdmf", line_mesh)
MPI.comm_world.barrier()
