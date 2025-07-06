from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Load mesh and facet tags
mesh = Mesh()
with XDMFFile(MPI.comm_world, "two_jets_mesh.xdmf") as xdmf:
    xdmf.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, "two_jets_facets.xdmf") as xdmf:
    xdmf.read(mvc, "facet_tags")

facet_tags = MeshFunction("size_t", mesh, mvc)

# Value that represents untagged facets
UNTAGGED = 18446744073709551615

# Loop over facets and get coordinates of untagged ones
mesh.init(1, 0)  # build connectivity from edges to vertices
V = mesh.coordinates()

fig, ax = plt.subplots()
for facet in facets(mesh):

    if facet.exterior():  # ✅ Only consider boundary facets
        tag = facet_tags[facet]

        if tag == UNTAGGED:
            x = [V[v.index()][0] for v in vertices(facet)]
            y = [V[v.index()][1] for v in vertices(facet)]
            print("Facet coords:", list(zip(x, y)))
            ax.plot(x, y, color="red", linewidth=2)

ax.set_title("Untagged Facets in Red")
ax.set_aspect("equal")
plt.show()
