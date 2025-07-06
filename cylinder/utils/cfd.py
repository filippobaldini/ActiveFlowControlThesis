from dolfin import *


# Set log level
set_log_level(LogLevel.WARNING)


T = 15.0
dtn = 0.0005
num_steps = int(T / dtn)
D = 0.5  # Cylinder diameter
NB_JETS = 3


visc = 5e-3  # dynamic viscosity
rho = 1  # density

mesh = Mesh()
with XDMFFile(MPI.comm_world, f"mesh/cylinder_{NB_JETS}jets.xdmf") as xdmf:
    xdmf.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, f"mesh/cylinder_{NB_JETS}jets_facets.xdmf") as xdmf:
    xdmf.read(mvc, "name_to_read")

surfaces = MeshFunction("size_t", mesh, mvc)


inlet_tag = 3
outlet_tag = 4
wall_tag1 = 1  # bottom
wall_tag2 = 2  # top
cylinder_tag = 5

jet_tags = list(range(cylinder_tag + 1, cylinder_tag + 1 + NB_JETS))


bot = mesh.coordinates().min(axis=0)[1]
top = mesh.coordinates().max(axis=0)[1]
# width of inlet channel
H = top - bot

# #
U_in = 1.5

# inflow_profile = Expression(
#     ("U_in * (1.0 - pow(2.0*(x[1] - H/2.0)/H, 2))", "0.0"),
#     U_in=U_in, H=H, degree=2
# )
inflow_profile = Expression(
    ("-4*U_in*(x[1]-bot)*(x[1]-top)/H/H", "0"),
    top=top,
    bot=bot,
    H=H,
    U_in=U_in,
    degree=2,
)

U_bar = (2 * U_in) / 3

Re = int(U_bar * rho * D / visc)

# Defining the functional spaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# #Define trial and test functions
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
# # Define functions for solutions at previous and current time steps
u_init = Function(V)
p_init = Function(Q)
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)
u_aux = Function(V)
# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
ix = Constant((1, 0))
k = Constant(dtn)
mu = Constant(visc)

# # Define boundary conditions
# # Inlet velocity
bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
# # No slip
bcu_wall1 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, wall_tag1)
bcu_wall2 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, wall_tag2)
bcu_wall3 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, cylinder_tag)


# # Fixing outflow pressure
bcp_outflow = DirichletBC(Q, Constant(0.0), surfaces, 4)

bcu_jets = [DirichletBC(V, Constant((0.0, 0.0)), surfaces, tag) for tag in jet_tags]
# bcu_jet = DirichletBC(V,g, surfaces ,6)

bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3] + bcu_jets

bcp = [bcp_outflow]

# Symmetrica part of the gradient
epsilon = lambda u: sym(nabla_grad(u))

# Cauchy stress tensor
sigma = lambda u, p: 2 * mu * epsilon(u) - p * Identity(2)

# Fractional step method Chorin Temam
# with consistency correction for pressure

# Define variational problem for step 1 (advection diffusion problem)
# u solution of step 1
# Define variational problem for step 1
F1 = (
    rho * dot((u - u_n) / k, v) * dx
    + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    + inner(sigma(U, p_n), epsilon(v)) * dx
    + dot(p_n * n, v) * ds
    - dot(mu * nabla_grad(U) * n, v) * ds
    - dot(f, v) * dx
)

# # Compute residual of the strong form of the momentum equation
# residual = rho*((u - u_n)/dtn + dot(u_n, nabla_grad(u_n))) - div(sigma(U, p_n)) - f

# # Characteristic element length (for simplicity, isotropic formulation)
# h = CellDiameter(mesh)

# # GLS stabilization parameter (can be tuned based on dt, mu, etc.)
# tau = (h**2) / (4.0 * mu)

# # Add GLS stabilization term
# gls_term = tau * dot(residual, dot(u_n, nabla_grad(v))) * dx
# F1 += gls_term

# # Constants
# delta = Constant(0.1)  # stabilization weight, tune as needed

# # Convecting velocity (from previous step)
# w = u_n

# # Cell size
# h = CellDiameter(mesh)

# # Viscosity and density
# nu = mu / rho

# # --- Residual of momentum equation (linearized around u_n) ---
# residual_supg = (
#     - nu * div(nabla_grad(u_n))                              # diffusion
#     + dot(w, nabla_grad(u_n))                                # advection
#     + 0.5 * div(w) * u_n                                     # weak compressibility
#     + grad(p_n)                                            # pressure gradient
#     - f                                                    # forcing
# )

# # --- Test function residual (SUPG projection) ---
# test_supg = dot(w, nabla_grad(v)) + 0.5 * div(v) * w

# # SUPG term
# F_supg = delta * inner(residual_supg, test_supg) * dx

# # --- Optional: Continuity stabilization (LSIC) ---
# F_divdiv = delta * inner(div(u_n), div(v)) * dx

# # Add to Step 1 variational form
# F1 += F_supg + F_divdiv

a1 = lhs(F1)
L1 = rhs(F1)
# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx

# Define variational problem for step 3
a3 = dot(u, v) * dx
L3 = dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]
[bc.apply(A3) for bc in bcu]

As = [A1, A2, A3]

# Create solvers
solvers = list(map(lambda x: LUSolver(), range(3)))

# Set matrices
for s, A in zip(solvers, As):
    s.set_operator(A)

import tqdm

timer = tqdm.tqdm("Solving forward problem", total=num_steps)

# outfile_u = XDMFFile(f"results/uRe{Re}.xdmf")
# outfile_p = XDMFFile(f"results/pRe{Re}.xdmf")

outfile_u_init = XDMFFile(f"mesh/init/uRe{Re}_init{NB_JETS}.xdmf")
outfile_p_init = XDMFFile(f"mesh/init/pRe{Re}_init{NB_JETS}.xdmf")


# Set parameters for better file output
# outfile_u.parameters["flush_output"] = True
# outfile_u.parameters["functions_share_mesh"] = True
# outfile_p.parameters["flush_output"] = True
# outfile_p.parameters["functions_share_mesh"] = True

outfile_u_init.parameters["flush_output"] = True
outfile_u_init.parameters["functions_share_mesh"] = True
outfile_p_init.parameters["flush_output"] = True
outfile_p_init.parameters["functions_share_mesh"] = True

# # Read the velocity and pressure data from XDMF files
# with XDMFFile("mesh/uRe1000_init.xdmf") as velocity_xdmf:
#     velocity_xdmf.read_checkpoint(u_init, 'velocity', 0)

# with XDMFFile("mesh/pRe1000_init.xdmf") as pressure_xdmf:
#     pressure_xdmf.read_checkpoint(p_init, 'pressure', 0)

u_n.assign(u_init)
p_n.assign(p_init)


# outfile_u.write_checkpoint(u_n, "velocity", 0, XDMFFile.Encoding.HDF5,append=True)
# outfile_p.write_checkpoint(p_n, "pressure", 0, XDMFFile.Encoding.HDF5,append=True)


for n_steps in range(num_steps):

    timer.update(1)
    t = n_steps * dtn

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solvers[0].solve(u_.vector(), b1)
    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solvers[1].solve(p_.vector(), b2)
    # Step 3: Velocity correction step
    b3 = assemble(L3)
    [bc.apply(b3) for bc in bcu]
    solvers[2].solve(u_.vector(), b3)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    # if n_steps == 9990:
    # if n_steps % 100 == 0:
    #     outfile_u.write_checkpoint(u_n, "velocity", n_steps, XDMFFile.Encoding.HDF5,append=True)
    #     outfile_p.write_checkpoint(p_n, "pressure", n_steps, XDMFFile.Encoding.HDF5,append=True)

    if n_steps == 29999:
        outfile_u_init.write_checkpoint(
            u_n, "velocity", n_steps, XDMFFile.Encoding.HDF5, append=True
        )
        outfile_p_init.write_checkpoint(
            p_n, "pressure", n_steps, XDMFFile.Encoding.HDF5, append=True
        )

# outfile_u.close()
# outfile_p.close()

outfile_u_init.close()
outfile_p_init.close()
