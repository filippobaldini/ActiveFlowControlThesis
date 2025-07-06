from dolfin import *


# Set log level
set_log_level(LogLevel.WARNING)


T = 10.0
dtn = 0.0005
num_steps = int(T / dtn)


visc = 6.66e-4  # dynamic viscosity
rho = 1  # density

mesh = Mesh()
with XDMFFile(MPI.comm_world, "mesh/our_mesh.xdmf") as xdmf:
    xdmf.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, "mesh/our_mesh_facets.xdmf") as xdmf:
    xdmf.read(mvc, "name_to_read")

surfaces = MeshFunction("size_t", mesh, mvc)


inlet_tag = 1
outlet_tag = 2
wall_tag1 = 3
wall_tag2 = 4
wall_tag3 = 5
jet1_tag = 6


bot = mesh.coordinates().min(axis=0)[1] + 0.1
top = mesh.coordinates().max(axis=0)[1]
# width of inlet channel
H = top - bot

# #
U_in = 1.5

inflow_profile = Expression(
    ("-4*U_in*(x[1]-bot)*(x[1]-top)/H/H", "0"),
    bot=bot,
    top=top,
    H=H,
    U_in=U_in,
    degree=2,
)


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
bcu_inlet = DirichletBC(V, inflow_profile, surfaces, 1)
# # No slip
bcu_wall1 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 3)
bcu_wall2 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 4)
bcu_wall3 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 5)


# # Fixing outflow pressure
bcp_outflow = DirichletBC(Q, Constant(0.0), surfaces, 2)

bcu_jet = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 6)

# bcu_jet = DirichletBC(V,g, surfaces ,6)

bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3, bcu_jet]

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

Re = int(rho * 0.1 / visc)

outfile_u = XDMFFile(f"mesh/init_2/u_Re{Re}.xdmf")
outfile_p = XDMFFile(f"mesh/init_2/p_Re{Re}.xdmf")


# Set parameters for better file output
outfile_u.parameters["flush_output"] = True
outfile_u.parameters["functions_share_mesh"] = True
outfile_p.parameters["flush_output"] = True
outfile_p.parameters["functions_share_mesh"] = True


u_n.assign(u_init)
p_n.assign(p_init)


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
    if n_steps == 19900:
        outfile_u.write_checkpoint(u_, "u0", 0, XDMFFile.Encoding.HDF5)
        outfile_p.write_checkpoint(p_, "p0", 0, XDMFFile.Encoding.HDF5)

outfile_u.close()
outfile_p.close()
