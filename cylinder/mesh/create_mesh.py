import gmsh
import numpy as np
from mpi4py import MPI

# Initialize Gmsh
gmsh.initialize()
proc = MPI.COMM_WORLD.rank

if proc == 0:
    gmsh.model.add("cylinder_2jets_105_255")

    # Parameters based on HydroGym's dimensions
    D = 0.5  # Cylinder diameter
    L_upstream = 3 * D
    L_downstream = 17 * D
    H_domain_up = 2.1 * D
    H_domain_down = 2 * D

    # Mesh size parameters
    cylinder_size = 0.03
    domain_size = 0.2
    inflow_size = 0.1

    # Define domain points
    points = [
        (-L_upstream, -H_domain_down, 0, inflow_size),
        (L_downstream, -H_domain_down, 0, domain_size),
        (L_downstream, H_domain_up, 0, domain_size),
        (-L_upstream, H_domain_up, 0, inflow_size),
    ]

    ptags = [gmsh.model.geo.addPoint(*p) for p in points]

    # Define outer rectangle lines
    lines = [
        gmsh.model.geo.addLine(ptags[0], ptags[1]),
        gmsh.model.geo.addLine(ptags[1], ptags[2]),
        gmsh.model.geo.addLine(ptags[2], ptags[3]),
        gmsh.model.geo.addLine(ptags[3], ptags[0]),
    ]

    jet_angles_deg = [0.0, 105.0, 255.0]  # jet angles
    jet_angles_rad = [
        np.radians(angle) for angle in jet_angles_deg
    ]  # Convert to radians
    width = np.pi / 18  # jet width

    jet_positioning = [
        [jet_angles - (width / 2), jet_angles + (width / 2)]
        for jet_angles in jet_angles_rad
    ]  # jet angles

    jet_points = []
    for i in range(0, len(jet_positioning)):
        jet_points.append(
            gmsh.model.geo.addPoint(
                (D / 2) * np.cos(jet_positioning[i][0]),
                (D / 2) * np.sin(jet_positioning[i][0]),
                0,
                cylinder_size,
            )
        )
        jet_points.append(
            gmsh.model.geo.addPoint(
                (D / 2) * np.cos(jet_positioning[i][1]),
                (D / 2) * np.sin(jet_positioning[i][1]),
                0,
                cylinder_size,
            )
        )

    # Create cylinder
    cylinder_center = gmsh.model.geo.addPoint(0, 0, 0, cylinder_size)

    circle_lines = []
    for i in range(0, len(jet_points) - 1):
        circle_lines.append((jet_points[i], jet_points[i + 1]))

    circle_lines.append((jet_points[-1], jet_points[0]))  # close the circle

    circle_tags = []

    for start, end in circle_lines:
        circle_tags.append(gmsh.model.geo.addCircleArc(start, cylinder_center, end))

    # Curve loops and surfaces
    domain_loop = gmsh.model.geo.addCurveLoop(lines)
    cylinder_loop = gmsh.model.geo.addCurveLoop(circle_tags)

    fluid_surface = gmsh.model.geo.addPlaneSurface([domain_loop, cylinder_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [lines[0]], 1)  # Bottom wall
    gmsh.model.setPhysicalName(1, 1, "bottom")

    gmsh.model.addPhysicalGroup(1, [lines[2]], 2)  # Top wall
    gmsh.model.setPhysicalName(1, 2, "top")

    gmsh.model.addPhysicalGroup(1, [lines[3]], 3)  # Inflow
    gmsh.model.setPhysicalName(1, 3, "inlet")

    gmsh.model.addPhysicalGroup(1, [lines[1]], 4)  # Outflow
    gmsh.model.setPhysicalName(1, 4, "outlet")

    cylinder_arcs = []
    for i, tag in enumerate(circle_tags):
        if i % 2 == 0:
            # Even index → jet arc
            gmsh.model.addPhysicalGroup(1, [tag], 6 + i // 2)
            gmsh.model.setPhysicalName(1, 6 + i // 2, f"jet{(i//2)+1}")
        else:
            cylinder_arcs.append(tag)

    gmsh.model.addPhysicalGroup(1, cylinder_arcs, 5)
    gmsh.model.setPhysicalName(1, 5, "cylinder")

    gmsh.model.addPhysicalGroup(2, [fluid_surface], 1)
    gmsh.model.setPhysicalName(2, 1, "fluid")

    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("cylinder_2jets_105_255.msh")

gmsh.finalize()
