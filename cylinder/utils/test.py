from FlowSolver import FlowSolver
import numpy as np
from dolfin import Expression

# Parametri minimi per costruire FlowSolver
geometry_params = {
    "num_control_jets": 3,
    "radius": 0.25,
    "center": [0.0, 0.0],
    "width": np.pi / 18,
}

flow_params = {
    "mu": 1e-3,
    "rho": 1.0,
    "U_in": 1.5,
    "inflow_profile": lambda mesh, degree: Expression(
        ("-4*U_in*(x[1]-bot)*(x[1]-top)/H/H", "0"),
        bot=mesh.coordinates().min(axis=0)[1] + 0.1,
        top=mesh.coordinates().max(axis=0)[1],
        H=mesh.coordinates().max(axis=0)[1] - (mesh.coordinates().min(axis=0)[1] + 0.1),
        U_in=1.5,
        degree=2,
    ),
    "u_init": "mesh/init/u_init.xdmf",  # dovrebbe combaciare con il tuo file .xdmf
    "p_init": "mesh/init/p_init.xdmf",
}

solver_params = {
    "dt": 0.0005,
    "solver_type": "lu",
    "preconditioner_step_1": "default",
    "preconditioner_step_2": "amg",
    "preconditioner_step_3": "jacobi",
    "la_solver_step_1": "gmres",
    "la_solver_step_2": "gmres",
    "la_solver_step_3": "cg",
}

# Istanzia il solver
solver = FlowSolver(flow_params, geometry_params, solver_params)

for stip in range(10):
    solver.evolve([0.0, 0.0, 0.0])
    Cd, Cl = solver.compute_drag_lift()
    print(f"{stip}\t{Cd:.4f}\t{Cl:.4f}")


# Ora attivo il getto 1
print("Step\tCd\tCl")
for step in range(20):
    solver.evolve([1.0, 0.0, 0.0])
    Cd, Cl = solver.compute_drag_lift()
    print(f"{step}\t{Cd:.4f}\t{Cl:.4f}")
