from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from matplotlib import pyplot

import os, sys
import moola


def sign(x):
    return conditional(gt(x, 0.0), 1.0, conditional(lt(x, 0.0), -1.0, 0.0))



# Set log level
set_log_level(LogLevel.WARNING)

# Next, we prepare the mesh,

#Importing mesh and vizualizing mesh
mesh = Mesh()
f    = HDF5File(mesh.mpi_comm(), 'mesh/our_mesh.h5','r')
f.read(mesh, 'mesh', False)
# plot(mesh)
# pyplot.show()
# and set a time step size:

# Choose a time step size
dtn = 5*1e-4
T = 0.5
num_steps = T/dtn

visc = 1E-3 #dynamic viscosity

rho = 1 #density

#Choose functional weights
beta1 = Constant(0.8) #vorticity term 
beta2 = Constant(0.2) #viscous dissipation term

alpha = Constant(0.3) #regularization

surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)    
f.read(surfaces, 'facet')                                           

# expression for the time dependent boundary term

class Jet(UserExpression):
    def __init__(self, omega=Constant(1e-1),frequency = Constant(1.), **kwargs):
        """ Construct the source function """
        super().__init__(self,**kwargs)
        self.t = 0.0
        self.omega = omega
        self.freq = frequency

    def eval(self, value, x):
        """ Evaluate the source function """
        
        value[1] = float(self.omega)*(x[0]-0.95)*(1-x[0])/0.0025 * (1+sin((self.freq)*2*pi*self.t))*0.5
        value[0] = Constant((0.0))
    

    def value_shape(self):
        return (2,)
    
    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value




class JetOmegaDerivative(UserExpression):
    def __init__(self, omega=Constant(1e-1),frequency = Constant(1.) ,Source=None, **kwargs):
        """ Construct the source function derivative """
        super().__init__(**kwargs)
        self.t = 0.0
        self.omega = omega
        self.freq = frequency
        self.source = Source  # needed to get the matching time instant

    def eval(self, value, x):
        """ Evaluate the source function's derivative """
        
        value[1] = (x[0]-0.95)*(1-x[0])/0.0025 * (1+sin(self.freq*2*pi*self.source.t))*0.5
        value[0] = Constant((0.0))
    
    def value_shape(self):
        return (2,)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value



class JetFrequencyDerivative(UserExpression):
    def __init__(self, omega=Constant(1e-1),frequency= Constant(1.), Source=None, **kwargs):
        """ Construct the source function derivative """
        super().__init__(**kwargs)
        self.t = 0.0
        self.omega = omega
        self.freq = frequency
        self.source = Source  # needed to get the matching time instant

    def eval(self, value, x):
        """ Evaluate the source function's derivative """
        
        value[1] = (float(self.omega)*(x[0]-0.95)*(1-x[0])/0.0025) * cos((self.freq)*2*pi*self.source.t) * 2*pi*self.source.t*0.5
        value[0] = Constant((0.0))
    
    def value_shape(self):
        return (2,)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value




# Before the inverse problem can be solved, we have to implement the forward problem:

def forward(Omega, Frequency, annotate=False, optimal=False):
    """ The forward problem """


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
    mu = Constant(visc)


     # Define the source expression with dependencies on Omega and Frequency
    jet = Jet(omega=Omega, frequency=Frequency, degree=2, name="source")

    # Provide dependencies for both Omega and Frequency
    jet.dependencies = [Omega, Frequency]

    # Define the derivative expressions
    omega_derivative = JetOmegaDerivative(omega=Omega, frequency=Frequency, Source=jet, degree=2, name="derivative_omega")
    frequency_derivative = JetFrequencyDerivative(omega=Omega, frequency=Frequency, Source=jet, degree=2, name="derivative_frequency")

    # Assign user-defined derivatives
    jet.user_defined_derivatives = {
        Omega: omega_derivative,
        Frequency: frequency_derivative
    }



     # # Define boundary conditions
    # # Inlet velocity
    bcu_inlet = DirichletBC(V, inflow_profile,surfaces,1) 
    # # No slip
    bcu_wall1 = DirichletBC(V, Constant((0., 0.)),surfaces, 3)
    bcu_wall2 = DirichletBC(V, Constant((0., 0.)),surfaces, 4)
    bcu_wall3 = DirichletBC(V, Constant((0., 0.)),surfaces, 5)
        
    # # Fixing outflow pressure
    bcp_outflow = DirichletBC(Q, Constant(0.),surfaces, 2)
    
    #Control boundary condition
    bcu_jet = DirichletBC(V, jet, surfaces , 6)


    bcu = [bcu_inlet,bcu_wall1,bcu_wall2,bcu_wall3,bcu_jet]

    bcp = [bcp_outflow]

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))
    # Define variational problem for step 1


    F1 = rho*dot((u - u_n) / k, v)*dx \
        + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(sigma(U, p_n), epsilon(v))*dx \
        + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)
    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
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


    if(optimal):
        outfile_u = XDMFFile("output/u_opt.xdmf")
        outfile_p = XDMFFile("output/p_opt.xdmf")
    else:
        outfile_u = XDMFFile("output/u.xdmf")
        outfile_p = XDMFFile("output/p.xdmf")

    # Read the velocity and pressure data from XDMF files
    with XDMFFile("mesh/u_init.xdmf") as velocity_xdmf:
        velocity_xdmf.read_checkpoint(u_init,'u0',0)

    with XDMFFile("mesh/p_init.xdmf") as pressure_xdmf:
        pressure_xdmf.read_checkpoint(p_init,'p0',0)

    if np.isnan(u_init.vector().norm('l2')):
        print("NaN detected in u_init")


    u_n.assign(u_init,annotate=annotate)
    p_n.assign(p_init,annotate=annotate)
    
    outfile_u.write(u_n,0)
    outfile_p.write(p_n,0)

    J = assemble(0.5 * beta1 * inner(curl(u_n), curl(u_n)) * dx +
                 0.5 * beta2 * inner(grad(u_n), grad(u_n)) * dx +
                 alpha * inner(u_n, u_n) * dx)

    # The actual timestepping    
    i = 1
    t = 0.0  # Initial time
    import tqdm
    timer = tqdm.tqdm("Solving forward problem", total=num_steps)

    while t < T - .5 * float(k):
        timer.update(1)
        
        jet.t = t + float(k)
        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        for bc in bcu:
            bc.apply(b1)        
        solvers[0].solve(u_.vector(), b1,annotate=annotate)
        # Step 2: Pressure correction step
        b2 = assemble(L2)
        for bc in bcp:
            bc.apply(b2)
        solvers[1].solve(p_.vector(), b2,annotate=annotate)
        # Step 3: Velocity correction step
        b3 = assemble(L3)
        for bc in bcu:
            bc.apply(b3)
        solvers[2].solve(u_.vector(), b3,annotate=annotate)
        # Update previous solution
        u_n.assign(u_,annotate=annotate)
        p_n.assign(p_,annotate=annotate)
        if np.isnan(u_n.vector().norm('l2')):
            print(f"NaN detected in u_n at time {t}")
            break

        outfile_u.write(u_n, i)
        outfile_p.write(p_n, i)
        J += assemble(0.5 * beta1 * inner(curl(u_n), curl(u_n)) * dx +
                 0.5 * beta2 * inner(grad(u_n), grad(u_n)) * dx +
                 alpha * inner(u_n, u_n) * dx)
        
        i += 1
        t = i * float(k)
        

    outfile_u.close()
    outfile_p.close()

    return J


def optimize(omega_init, frequency_init):
    """The optimization routine to control both omega and frequency."""

    # Define the control variables as Constants
    Omega = Constant(omega_init)
    Frequency = Constant(frequency_init)

    # Define the functional
    J = forward(Omega, Frequency, annotate=True, optimal=False)
    print("Functional Value = ", float(J))

    controls = [Control(Omega), Control(Frequency)]

    # Define the reduced functional
    Jhat = ReducedFunctional(J, controls)

    omega_opt,freq_opt = minimize(Jhat, method="L-BFGS-B",
                            tol=1.0e-12, options={"disp": True, "gtol": 1.0e-12})
    
    print("Optimal Omega value = ", float(omega_opt))
    print("Optimal Frequency value = ", float(freq_opt))

    # Return the optimized values
    return omega_opt, freq_opt

# Lastly we implement some code to run the optimization:
    
omega = Constant(1e-6)
frequency = Constant(1.)

omega_opt,freq_opt = optimize(omega,frequency)

forward(omega_opt,freq_opt,False,True)