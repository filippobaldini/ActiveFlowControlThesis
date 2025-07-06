import gmsh
import numpy as np
from mpi4py import MPI

gmsh.initialize()
proc = MPI.COMM_WORLD.rank

if proc == 0:
    gmsh.model.add("two_jets")

    # Parameters
    inf_width = 0.2
    step_height = 0.1
    step_dist = 1.0
    top_len = 3.0
    control_width = 0.02

    # Mesh sizes
    inflow_size = 0.02
    control_size = 0.004
    step_size = 0.016
    outflow_size = 0.03

    # Derived parameters
    outflow_width = step_height + inf_width

    # Points
    points = [
        (0, step_height, 0, inflow_size),  # p1
        (step_dist, step_height, 0, step_size),  # p2
        (step_dist, (step_height * 3 / 4) + control_width, 0, control_size),  # p3
        (step_dist, step_height * 3 / 4, 0, control_size),  # p4
        (step_dist, step_height * 1 / 4, 0, control_size),  # p5
        (step_dist, (step_height * 1 / 4) - control_width, 0, control_size),  # p6
        (step_dist, 0, 0, step_size),  # p7
        (top_len, 0, 0, outflow_size),  # p8
        (top_len, outflow_width, 0, outflow_size),  # p9
        (0, outflow_width, 0, inflow_size),  # p10
    ]

    ptags = []
    for i, (x, y, z, s) in enumerate(points):
        ptags.append(gmsh.model.geo.addPoint(x, y, z, s))

    # Lines
    lines = [
        (ptags[0], ptags[1]),  # l0
        (ptags[1], ptags[2]),  # l1
        (ptags[2], ptags[3]),  # l2
        (ptags[3], ptags[4]),  # l3
        (ptags[4], ptags[5]),  # l4
        (ptags[5], ptags[6]),  # l5
        (ptags[6], ptags[7]),  # l6
        (ptags[7], ptags[8]),  # l7
        (ptags[8], ptags[9]),  # l8
        (ptags[9], ptags[0]),  # l9
    ]

    ltags = []
    for start, end in lines:
        ltags.append(gmsh.model.geo.addLine(start, end))

    print(f"Total ltags: {len(ltags)}")
    print(
        "Tagged lines:",
        sorted(
            set(
                [
                    ltags[0],
                    ltags[1],
                    ltags[2],
                    ltags[3],
                    ltags[4],
                    ltags[5],
                    ltags[6],
                    ltags[7],
                    ltags[8],
                    ltags[9],
                ]
            )
        ),
    )

    # Curve loop and surface
    loop = gmsh.model.geo.addCurveLoop(ltags)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    all_lines = gmsh.model.getEntities(dim=1)
    print(f"Total lines in geometry: {len(all_lines)}")

    # Physical Groups
    gmsh.model.addPhysicalGroup(1, [ltags[9]], 1)  # Inlet
    gmsh.model.setPhysicalName(1, 1, "Inflow")

    gmsh.model.addPhysicalGroup(1, [ltags[7]], 2)  # Outlet
    gmsh.model.setPhysicalName(1, 2, "Outflow")

    gmsh.model.addPhysicalGroup(1, [ltags[0]], 3)  # Before Step
    gmsh.model.setPhysicalName(1, 3, "Wall1")

    gmsh.model.addPhysicalGroup(1, [ltags[1]], 4)  # Before Jet 1
    gmsh.model.setPhysicalName(1, 4, "Wall2")

    gmsh.model.addPhysicalGroup(1, [ltags[3]], 5)  # Before Jet 2
    gmsh.model.setPhysicalName(1, 5, "Wall3")

    gmsh.model.addPhysicalGroup(1, [ltags[5]], 8)  # Before bottom wall
    gmsh.model.setPhysicalName(1, 8, "Wall4")

    gmsh.model.addPhysicalGroup(1, [ltags[6]], 9)  # Bottom wall
    gmsh.model.setPhysicalName(1, 9, "Wall5")

    gmsh.model.addPhysicalGroup(1, [ltags[8]], 10)  # Top wall
    gmsh.model.setPhysicalName(1, 10, "Wall6")

    gmsh.model.addPhysicalGroup(1, [ltags[2]], 6)  # Jet1
    gmsh.model.setPhysicalName(1, 6, "Jet1")

    gmsh.model.addPhysicalGroup(1, [ltags[4]], 7)  # Jet2
    gmsh.model.setPhysicalName(1, 7, "Jet2")

    gmsh.model.addPhysicalGroup(2, [surface], 5)  # Fluid domain
    gmsh.model.setPhysicalName(2, 5, "Fluid")

    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("two_jets.msh")


gmsh.finalize()
