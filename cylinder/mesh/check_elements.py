# from dolfin import *

# # Load mesh (assuming it's in XDMF or MSH format depending on how you converted it)
# mesh = Mesh()
# with XDMFFile(MPI.comm_world, "cylinder_3jets.xdmf") as xdmf:
#     xdmf.read(mesh)

# print("Number of nodes (vertices):", mesh.num_vertices())
# print("Number of triangular elements (cells):", mesh.num_cells())


from dolfin import *

filename = "cylinder_3jets.h5"
h5 = HDF5File(MPI.comm_world, filename, "r")
print(h5.has_dataset("mesh"))
print(h5.has_dataset("facet"))  # or try "boundaries"
