import gmsh
import numpy as np
from mpi4py import MPI

# User-defined parameters
jet_angles_deg = [105.0, 255.0]  # Jet positions in degrees
D = 0.5  # Cylinder diameter
width = np.pi / 18  # Jet width in radians

# Initialize Gmsh
gmsh.initialize()
proc = MPI.COMM_WORLD.rank

if proc == 0:
    gmsh.model.add("cylinder_jets")

    # Domain dimensions
    L_upstream = 3 * D
    L_downstream = 17 * D
    H_domain_up = 2.1 * D
    H_domain_down = 2 * D

    # Mesh size
    cylinder_size = 0.03
    domain_size = 0.2
    inflow_size = 0.1

    # Outer rectangle
    points = [
        (-L_upstream, -H_domain_down, 0, inflow_size),
        (L_downstream, -H_domain_down, 0, domain_size),
        (L_downstream, H_domain_up, 0, domain_size),
        (-L_upstream, H_domain_up, 0, inflow_size),
    ]
    ptags = [gmsh.model.geo.addPoint(*p) for p in points]
    lines = [
        gmsh.model.geo.addLine(ptags[0], ptags[1]),
        gmsh.model.geo.addLine(ptags[1], ptags[2]),
        gmsh.model.geo.addLine(ptags[2], ptags[3]),
        gmsh.model.geo.addLine(ptags[3], ptags[0]),
    ]

    # Cylinder and jet arc setup
    R = D / 2
    center = gmsh.model.geo.addPoint(0, 0, 0, cylinder_size)
    jet_angles_rad = [np.radians(a) for a in jet_angles_deg]

    # Define jet start/end angles
    jet_intervals = [(a - width / 2, a + width / 2) for a in jet_angles_rad]

    # All arc endpoints
    arc_angles = []
    for interval in jet_intervals:
        arc_angles.extend(interval)
    arc_angles = np.mod(arc_angles, 2 * np.pi)
    arc_angles = np.unique(np.sort(arc_angles))

    # Add arc points
    arc_points = [
        gmsh.model.geo.addPoint(R * np.cos(a), R * np.sin(a), 0, cylinder_size)
        for a in arc_angles
    ]

    # Create arcs
    arcs = []
    arc_angle_vals = list(arc_angles)
    for i in range(len(arc_points)):
        a_start = arc_angle_vals[i]
        a_end = arc_angle_vals[(i + 1) % len(arc_points)]

        # Ensure proper CCW arc direction
        if a_end < a_start:
            a_end += 2 * np.pi
        a_mid = 0.5 * (a_start + a_end)
        a_mid = np.mod(a_mid, 2 * np.pi)

        is_jet = any(
            start <= a_mid <= end or (start > end and (a_mid >= start or a_mid <= end))
            for (start, end) in jet_intervals
        )

        tag = gmsh.model.geo.addCircleArc(
            arc_points[i], center, arc_points[(i + 1) % len(arc_points)]
        )
        arcs.append((tag, is_jet))

    # Build loops and surfaces
    domain_loop = gmsh.model.geo.addCurveLoop(lines)
    cylinder_loop = gmsh.model.geo.addCurveLoop([a[0] for a in arcs])
    fluid_surface = gmsh.model.geo.addPlaneSurface([domain_loop, cylinder_loop])

    gmsh.model.geo.synchronize()

    # Tag boundaries
    gmsh.model.addPhysicalGroup(1, [lines[0]], 1)
    gmsh.model.setPhysicalName(1, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [lines[2]], 2)
    gmsh.model.setPhysicalName(1, 2, "top")
    gmsh.model.addPhysicalGroup(1, [lines[3]], 3)
    gmsh.model.setPhysicalName(1, 3, "inlet")
    gmsh.model.addPhysicalGroup(1, [lines[1]], 4)
    gmsh.model.setPhysicalName(1, 4, "outlet")

    # Tag jet and cylinder arcs
    jet_id = 6
    cylinder_arcs = []
    for tag, is_jet in arcs:
        if is_jet:
            gmsh.model.addPhysicalGroup(1, [tag], jet_id)
            gmsh.model.setPhysicalName(1, jet_id, f"jet{jet_id - 5}")
            jet_id += 1
        else:
            cylinder_arcs.append(tag)

    gmsh.model.addPhysicalGroup(1, cylinder_arcs, 5)
    gmsh.model.setPhysicalName(1, 5, "cylinder")
    gmsh.model.addPhysicalGroup(2, [fluid_surface], 1)
    gmsh.model.setPhysicalName(2, 1, "fluid")

    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("cylinder_custom_jets.msh")

gmsh.finalize()
