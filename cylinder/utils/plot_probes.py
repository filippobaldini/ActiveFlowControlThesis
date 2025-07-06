from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Load mesh
mesh = Mesh()
with XDMFFile(MPI.comm_world, "mesh/cylinder_3jets.xdmf") as xdmf:
    xdmf.read(mesh)

# Define probe positions manually (as in your script)
center = np.array([0.0, 0.0])
radii = [0.3, 0.4]
num_probes_per_ring = 32
list_position_probes = []

# Circular rings
for r in radii:
    angles = np.linspace(0, 2 * np.pi, num_probes_per_ring, endpoint=False)
    for theta in angles:
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        list_position_probes.append(np.array([x, y]))

# Grid in front of cylinder
x_coords = np.linspace(-0.15, 0.27, 6)
y_coords = np.linspace(-0.6, 0.6, 7)
for x in x_coords:
    for y in y_coords:
        if (x**2 + y**2) > 0.6**2:
            list_position_probes.append(np.array([x, y]))

# Wake probes downstream
x_coords = np.linspace(0.3, 3.3, 10)
y_coords = np.linspace(-0.6, 0.6, 7)
for x in x_coords:
    for y in y_coords:
        list_position_probes.append(np.array([x, y]))

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
