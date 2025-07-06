from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Load mesh
mesh = Mesh()
with XDMFFile(MPI.comm_world, "mesh/our_mesh.xdmf") as xdmf:
    xdmf.read(mesh)

# initialization of the list containing the coordinates of the probes
list_position_probes = []
# collocate the probes in the more critical region for the recirculation area:
# that is the area below the step.
# It would be likely a good possible improvement to place some probes also in the upper area
positions_probes_for_grid_x = np.linspace(1, 2, 27)[1:-1]
positions_probes_for_grid_y = np.linspace(0, 0.1, 6)[1:-1]


for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))


probes = np.array(list_position_probes)
print("Nb of probes: ", len(probes))

# Plot
plt.figure(figsize=(14, 3))
plot(mesh, linewidth=0.2, color="gray")
plt.scatter(probes[:, 0], probes[:, 1], c="red", s=8, zorder=10)
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.show()
