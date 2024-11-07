import mshr
from dolfin import *
import numpy as np
from dolfin_adjoint import *
from matplotlib import pyplot, rc
import moola

# Set log level
set_log_level(LogLevel.WARNING)


# Parameters
T = 2.  # Total simulation time
dtn = 0.0005  # Time step size
num_steps = int(T / dtn)

viscosity = 0.001 #dynamic viscosity
rho = 1  # density
Re=70 #Reynold's number


#Importing mesh
mesh = Mesh()
f    = HDF5File(mesh.mpi_comm(), 'mesh/our_mesh.h5','r')
f.read(mesh, 'mesh', False)

plot(mesh)
pyplot.show()


# function that extracts the entities of dimension equal to the one of the 
# topology minus one from a mesh
surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)     

# apply mesh function to extract facets from mesh and store them into surfaces
f.read(surfaces, 'facet')                                           


bot = mesh.coordinates().min(axis=0)[1]+0.1
top = mesh.coordinates().max(axis=0)[1]
# width of inlet channel
H = top - bot

# # 
U_in = 1.5

inflow_profile = Expression(('-4*U_in*(x[1]-bot)*(x[1]-top)/H/H' , '0'), bot=bot, top=top, H=H, U_in=U_in, degree=2)


#Defining the functional spaces
Q = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, 'CG', 2)


# Class for the bd term
class Jet(UserExpression):
    def __init__(self, omega = Constant(1e-4), **kwargs):
        super().__init__(**kwargs)
        self.t = 0.0
        self.omega = omega

    def eval(self, value, x):

        # Evaluate the boundary condition
        value[1] = self.omega * (x[0] - 0.95) * (1 - x[0]) / 0.0025 * abs(sin(2 * pi * self.t))
        value[0] = 0.0

    def value_shape(self):
        return (2,)

class JetDerivative(UserExpression):
    def __init__(self,Omega,Jet=None, **kwargs):
        super().__init__(**kwargs)
        self.jet = Jet
        self.t = 0.0

    def eval(self, value, x):
        # Check if current time corresponds to this control variable
        value[1] = (x[0] - 0.95) * (1 - x[0]) / 0.0025 * abs(sin(2 * pi * self.jet.t))
        value[0] = 0.0


    def value_shape(self):
        return (2,)




def solve_cfd(Omega, annotate=False, opt=False):

    # Define trial and test functions
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    # Define functions for solutions at previous and current time steps
    u_n = Function(V, annotate=annotate)
    u_ = Function(V, annotate=annotate)
    p_n = Function(Q, annotate=annotate)
    p_ = Function(Q, annotate=annotate)
    u_init = Function(V, annotate=annotate)
    p_init = Function(Q, annotate=annotate)

    # Define expressions used in variational forms
    U = 0.5 * (u_n + u)
    n = FacetNormal(mesh)
    f = Constant((0, 0))
    k = Constant(dtn)
    mu = Constant(viscosity)

    # Initialize the Jet expression
    jet = Jet(Omega, degree=2, name="source")
    # Provide the coefficient on which this expression depends and its derivative
    jet.dependencies = [Omega]
    jet.user_defined_derivatives = {Omega: JetDerivative(Omega, jet, degree=2)}

    # Define boundary conditions
    bcu_inlet = DirichletBC(V, inflow_profile, surfaces, 1)
    bcu_wall1 = DirichletBC(V, Constant((0., 0.)), surfaces, 3)
    bcu_wall2 = DirichletBC(V, Constant((0., 0.)), surfaces, 4)
    bcu_wall3 = DirichletBC(V, Constant((0., 0.)), surfaces, 5)
    bcp_outflow = DirichletBC(Q, Constant(0.), surfaces, 2)
    bcu_jet = DirichletBC(V, jet, surfaces, 6)

    bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3, bcu_jet]
    bcp = [bcp_outflow]

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2 * mu * epsilon(u) - p * Identity(len(u))



    # Define variational problem for step 1
    F1 = rho * dot((u - u_n) / k, v) * dx \
        + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx \
        + inner(sigma(U, p_n), epsilon(v)) * dx \
        + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds \
        - dot(f, v) * dx
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

    As = [A1,A2,A3]

    # Create solvers
    solvers = list(map(lambda x: LUSolver(), range(3)))


    # Set matrices
    for s, A in zip(solvers, As):
        s.set_operator(A)

    import tqdm
    timer = tqdm.tqdm("Solving forward problem", total=num_steps)
    if opt:
        outfile_u = XDMFFile("output/u_opt.xdmf")
        outfile_p = XDMFFile("output/p_opt.xdmf")
    else:
        outfile_u = XDMFFile("output/u.xdmf")
        outfile_p = XDMFFile("output/p.xdmf")

    # Read the velocity and pressure data from XDMF files
    with XDMFFile("mesh/u_init.xdmf") as velocity_xdmf:
        velocity_xdmf.read_checkpoint(u_init, 'u0', 0)

    with XDMFFile("mesh/p_init.xdmf") as pressure_xdmf:
        pressure_xdmf.read_checkpoint(p_init, 'p0', 0)

    u_n.assign(u_init, annotate=annotate)
    p_n.assign(p_init, annotate=annotate)
    outfile_u.write(u_n, 0.0)
    outfile_p.write(p_n, 0.0)

    beta1 = 0.8
    beta2 = 0.2
    alpha = 0.5

    J = assemble(0.5 * beta1 * inner(curl(u_n), curl(u_n)) * dx +
                 0.5 * beta2 * inner(grad(u_n), grad(u_n)) * dx +
                 alpha * inner(u_n, u_n) * dx)

    for n_steps in range(num_steps):
        
        timer.update(1)
        t = n_steps * dtn
        jet.t = t  # Update time in jet expression

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solvers[0].solve(u_.vector(), b1,annotate=annotate)
        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solvers[1].solve(p_.vector(), b2,annotate=annotate)
        # Step 3: Velocity correction step
        b3 = assemble(L3)
        [bc.apply(b3) for bc in bcu]
        solvers[2].solve(u_.vector(), b3,annotate=annotate)
        
        # # Compute and print L2 norm of u_
        # u_L2 = sqrt(assemble(dot(u_, u_) * dx))
        # print(f"Time step {n_steps}, u_ L2 norm: {u_L2:.6e}")
        
        # Update previous solution
        u_n.assign(u_, annotate=annotate)
        p_n.assign(p_, annotate=annotate)
        outfile_u.write(u_, t)
        outfile_p.write(p_, t)
        Jtemp = assemble(0.5 * beta1 * inner(curl(u_), curl(u_)) * dx +
                         0.5 * beta2 * inner(grad(u_), grad(u_)) * dx +
                         alpha * inner(u_, u_) * dx)
        J += Jtemp

    outfile_u.close()
    outfile_p.close()

    return J

def optimize(Omega):


    # Define the control
    J = solve_cfd(Omega,True,False)

    print("Functional Value = ",float(J))
    
    # Define the control
    m = Control(Omega)

    dJd0 = compute_gradient(J,m)
    print("gradient = ", float(dJd0))

    Jhat = ReducedFunctional(J,m)

    omega_opt = minimize(Jhat, method="L-BFGS-B",
                            tol=1.0e-12, options={"disp": True, "gtol": 1.0e-12})
    
    # problem = MoolaOptimizationProblem(Jhat)
    # m_moola = moola.DolfinPrimalVector(Omega)
    # optimizer = moola.BFGS(problem, m_moola, 
    #         options={
    #     "tol": 1.0e-8,      # Tolerance for convergence
    #     "max_iter": 1000,   # Maximum number of iterations
    #     "print_level": 5    # Verbosity level
    # })

    # sol = optimizer.solve()
    # omega_opt = sol['control'].data

    # Print the obtained optimal value for the controls
    print("omega = %f" % float(omega_opt))
    
    return omega_opt


Omega_value= Constant(1e-6)  # Initial guess for Omega


omega_opt = optimize(Omega_value)


J = solve_cfd(omega_opt,False,True)
