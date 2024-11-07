from dolfin import * 
import numpy as np 
from control_bcs import JetBCValue
from probes import PressureProbeValues,VelocityProbeValues,TotalrecirculationArea


class FlowSolver(object):
    '''
    We here implement the fenics based CFD simulation used as episode template for learning and to 
    generate the baseline from which every episode starts
    '''

    def __init__(self, flow_params, geometry_params, solver_params):    

        self.T = 2.0
       
        # dynamic viscosity                                                                       	
        mu = Constant(flow_params['mu'])      

        # density    
        rho = Constant(flow_params['rho'])            
        
        #Importing mesh
        mesh = Mesh()
        f    = HDF5File(mesh.mpi_comm(), 'mesh/our_mesh.h5','r')
        f.read(mesh, 'mesh', False)

        # plot(mesh)
        # pyplot.show()


        # function that extracts the entities of dimension equal to the one of the 
        # topology minus one from a mesh
        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)     

        # apply mesh function to extract facets from mesh and store them into surfaces
        f.read(surfaces, 'facet')

        # These boundary condition tags should be hardcoded by gmsh during generation
        inlet_tag = 1
        outlet_tag = 2
        wall_tag1 = 3 
        # bottom before jet
        wall_tag2 = 4
        # step and bottom after step
        wall_tag3 = 5 
        # top wall
        control_tag = 6                                                                    
                                                                        

        # Define function spaces for velocity and pressure
        # continuous galerkin polynomials of degreee 2 and 1 respectively
        V = VectorFunctionSpace(mesh, 'CG', 2)       
        Q = FunctionSpace(mesh, 'CG', 1)                   

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)
        # functions for explicit terms
        u_n, p_n = Function(V), Function(Q)

        # External clock
        gtime = 0.                                                                 
                                                                        
        # Initialize u_n and p_n from u_init and p_init xdmf files where we have last step of baseline
        # simulation
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):
            if path in flow_params:
                comm = mesh.mpi_comm()
                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)   

        # Functions for solution at each step
        u_, p_ = Function(V), Function(Q)  

        # Temporal step
        dt = Constant(solver_params['dt'])

        # Define expressions used in variational forms

        U  = Constant(0.5)*(u_n + u)
        
        # Normal versor exiting any cell
        n  = FacetNormal(mesh) 

        # Forcing homogeneous term
        f  = Constant((0, 0))

        # Symmetrica part of the gradient
        epsilon = lambda u :sym(nabla_grad(u))                  

        # Cauchy stress tensor
        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)   

        # Fractional step method Chorin Temam
        # with consistency correction for pressure

        # Define variational problem for step 1 (advection diffusion problem)          
        # u solution of step 1
        F1 = (rho*dot((u - u_n) / dt, v)*dx                   # time derivative
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx      # transport non linear term trated implicitely
              + inner(sigma(U, p_n), epsilon(v))*dx           # diffusion semi-implicit with pressure correction
              + dot(p_n*n, v)*ds                              # Neumann pressure bcs
              - dot(mu*nabla_grad(U)*n, v)*ds                 # Neumann velocity bcs
              - dot(f, v)*dx)                                 # forcing term 
        

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2 (laplacian problem for pressure)                       
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx                                   
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3 (projection step)
        a3 = dot(u, v)*dx
        # u_ here will be the solution of the step 1 and the solution of this step will
        # be stored again in u_
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx
        
        # extract boundary condition at the inlet in closed form
        inflow_profile = flow_params['inflow_profile'](mesh, degree=2) 

        # Define boundary conditions, first those that are constant in time

        # Inlet velocity
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag) 
        # No slip
        bcu_wall1 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag1)
        bcu_wall2 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag2)
        bcu_wall3 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag3)
    
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)   

        # Now the expression for the control boundary condition

        # Parameters needed to initialize JetBCValue object         
        length_before_control = geometry_params['length_before_control']
        control_width = geometry_params['control_width']  
        step_height = geometry_params['step_height']
        jet_amplitude_tuning = geometry_params['tuning_parameters'] [0]
        frequency_amplitude_tuning = geometry_params['tuning_parameters'][1]
        frequency_shift_tuning = geometry_params['tuning_parameters'][2]

        jet_tag = control_tag                                                                                  
        
        self.jet = JetBCValue(gtime, length_before_control, step_height, control_width,  jet_amplitude_tuning, frequency_amplitude_tuning, frequency_shift_tuning, frequency=0, Q=0, degree=1)

        # Boundary condition for jet, here set as no-slip
        bcu_jet = DirichletBC(V, self.jet, surfaces, jet_tag)
    

        # All bcs objects together (where we don't impose anything we have homogeneous Neumann)
        # velocity bcs
        bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3, bcu_jet]
        # pressure bcs
        bcp = [bcp_outflow]
        
        # Initialize matrices for algebraic Chorin Temam
        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get('solver_type')
        assert solver_type in ('lu', 'la_solve')
        la_solver1 = solver_params.get('la_solver_step_1')
        la_solver2 = solver_params.get('la_solver_step_2')
        la_solver3 = solver_params.get('la_solver_step_3')
        precond1 = solver_params.get('preconditioner_step_1')
        precond2 = solver_params.get('preconditioner_step_2')
        precond3 = solver_params.get('preconditioner_step_3')
        
        # diret solver
        if solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver(), range(3)))

        # iterative solver
        else:
            # we have to afind reasonable preconditioners
            solvers = [KrylovSolver(la_solver1, precond1), 
                       KrylovSolver(la_solver2, precond2),
                       KrylovSolver(la_solver3, precond3)]

        # Set matrices
        for s, A in zip(solvers, As):
            s.set_operator(A)
            # set parameters for iterative method
            if not solver_type == 'lu':
                s.parameters['relative_tolerance'] = 1E-8
                s.parameters['monitor_convergence'] = False

        
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n

        # Rename u_, p_ with standard names
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.jet_tag = jet_tag  
  


    def evolve(self, jet_bc_values):
        '''
        Make one time step dt with the given values of jet boundary conditions
        '''
        # Update jet amplitude and frequency
        self.jet.Q = jet_bc_values[0]
        self.jet.freq = jet_bc_values[1]

        # Increments time
        self.gtime += self.dt(0) 

        self.jet.time = self.gtime

        # Updating inflow profile if it's function of time
        inflow = self.inflow_profile
        if hasattr(inflow, 'time'):
            inflow.time = self.gtime

        # solvers from the object
        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        # solving the 3 steps respectively in u_ p_ and u_ again
        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            solver.solve(uh.vector(), b)

        # updating last step parameters
        u_n.assign(u_)
        p_n.assign(p_)

        # return next step solution
        return u_, p_


    def get_observation(self, env):
        probes_value = env.ann_probes.sample(self.u_, self.p_)
        print("probes_value type:", type(probes_value))
        print("probes_value shape:", np.shape(probes_value))
        obs = np.array(probes_value).flatten()
        print("obs shape:", obs.shape)
        return obs


# function to impose inflow boundary condition
# parabolic profile normalized with maximum velocity U_in

def profile(mesh, degree):

    bot = mesh.coordinates().min(axis=0)[1]+0.1
    top = mesh.coordinates().max(axis=0)[1]

    # width of inlet channel
    H = top - bot

    # 
    U_in = 1.5

    return Expression(('-4*U_in*(x[1]-bot)*(x[1]-top)/H/H',
                    '0'), bot=bot, top=top, H=H, U_in=U_in, degree=degree)
