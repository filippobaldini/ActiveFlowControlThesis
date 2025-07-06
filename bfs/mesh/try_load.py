from dolfin import *
from collections import Counter

# Load the mesh and boundary markers
mesh = Mesh("two_jets.xml")
boundaries = MeshFunction("size_t", mesh, "two_jets_facet_region.xml")

# Print all unique tags on the boundary
print("Unique boundary tags:", set(boundaries.array()))

# Count how many facets have each tag
print("Facet tag counts:", Counter(boundaries.array()))
