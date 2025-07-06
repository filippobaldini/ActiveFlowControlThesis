import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

# Load mesh
mesh = Mesh()
with XDMFFile(MPI.comm_world, "mesh/cylinder_3jets.xdmf") as xdmf:
    xdmf.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, "mesh/cylinder_3jets_facets.xdmf") as xdmf:
    xdmf.read(mvc, "name_to_read")

surfaces = MeshFunction("size_t", mesh, mvc)
cylinder_tag = 5

point = Point(0.25, 0.0)  # Point to check
n = FacetNormal(mesh)

# Find the closest boundary facet around the point
closest_n = None
closest_d = 1e6

for facet in facets(mesh):
    mp = facet.midpoint()
    dist = mp.distance(point)
    if dist < closest_d and facet.exterior():  # only exterior facets
        closest_d = dist
        closest_n = n(mp)

print(f"Closest normal to point {point}: {closest_n}")
