from dolfin import *
import os

# Parameters
visc = 1e-3
rho = 1.0
D = 1.0
U_in = 0.5

# Load mesh and facet markers
mesh = Mesh()
with XDMFFile(MPI.comm_world, "mesh/cylinder_5jets.xdmf") as xdmf:
    xdmf.read(mesh)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, "mesh/cylinder_5jets_facets.xdmf") as xdmf:
    xdmf.read(mvc, "name_to_read")

surfaces = MeshFunction("size_t", mesh, mvc)

# Domain size for inlet profile
bot = mesh.coordinates().min(axis=0)[1]
top = mesh.coordinates().max(axis=0)[1]
H = top - bot

# Define inflow profile (parabolic)
inflow_profile = Expression(
    ("-4*U_in*(x[1]-bot)*(x[1]-top)/H/H", "0"),
    U_in=U_in,
    H=H,
    degree=2,
    bot=bot,
    top=top,
)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# # Define boundary conditions
# # Inlet velocity
bcu_inlet = DirichletBC(V, inflow_profile, surfaces, 3)
# # No slip
bcu_wall1 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 1)
bcu_wall2 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 2)
bcu_wall3 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 5)


# # Fixing outflow pressure
bcp_outflow = DirichletBC(Q, Constant(0.0), surfaces, 4)

bcu_jet = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 6)
bcu_jet1 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 7)
bcu_jet2 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 8)
bcu_jet3 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 9)
bcu_jet4 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, 10)

# bcu_jet = DirichletBC(V,g, surfaces ,6)

bcu = [
    bcu_inlet,
    bcu_wall1,
    bcu_wall2,
    bcu_wall3,
    bcu_jet,
    bcu_jet1,
    bcu_jet2,
    bcu_jet3,
    bcu_jet4,
]

bcp = [bcp_outflow]

# Define variational forms (steady Stokes)
mu = Constant(visc)
f = Constant((0.0, 0.0))

# a1 = inner(mu * nabla_grad(u), nabla_grad(v)) * dx - div(v) * p * dx
# a2 = -div(u) * q * dx
# L = dot(f, v) * dx


p_dummy = Function(Q)  # Just a placeholder

a1 = mu * inner(nabla_grad(u), nabla_grad(v)) * dx
L1 = dot(f, v) * dx

u_sol = Function(V)
solve(a1 == L1, u_sol, bcu)

a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_dummy), nabla_grad(q)) * dx - div(u_sol) * q * dx

p_sol = Function(Q)
solve(a2 == L2, p_sol, bcp)

# Save
os.makedirs("mesh/init", exist_ok=True)
with XDMFFile("mesh/init/u_stokes.xdmf") as xf:
    xf.write(u_sol)
with XDMFFile("mesh/init/p_stokes.xdmf") as pf:
    pf.write(p_sol)

print("✅ Stokes solution written to mesh/init/u_stokes.xdmf and p_stokes.xdmf")
