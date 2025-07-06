from dolfin import *
import numpy as np
from env.control_bcs import JetBCValue, WallJetBCValue


class FlowSolver(object):
    """
    We here implement the fenics based CFD simulation used as episode template for learning and to
    generate the baseline from which every episode starts
    """

    def __init__(self, flow_params, geometry_params, solver_params):


        import os

        # All'inizio del file, ad esempio in __init__ o dove definisci la mesh
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mesh_file = os.path.join(base_path, "mesh", "our_mesh.xdmf")
        facet_file = os.path.join(base_path, "mesh", "our_mesh_facets.xdmf")

        self.T = 2.0

        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params

        if self.flow_params["parametric"]:

            self.mu_list = flow_params["mu_list"]
            self.mu = np.random.choice(self.mu_list)

            self.Re = int(0.1 / self.mu)

            self.flow_params["u_init"] = f"mesh/init/u_Re{self.Re}.xdmf"
            self.flow_params["p_init"] = f"mesh/init/p_Re{self.Re}.xdmf"

            print(f"Selected mu: {self.mu}, Re: {self.Re}")
            print(f"Using initial conditions: u_Re{self.Re}.xdmf, p_Re{self.Re}.xdmf")

        else:
            self.mu = flow_params["mu"]
            self.Re = int(0.1 / self.mu)

            self.flow_params["u_init"] = os.path.join(base_path, "mesh", "init", f"u_Re{self.Re}.xdmf")
            self.flow_params["p_init"] = os.path.join(base_path, "mesh", "init", f"p_Re{self.Re}.xdmf")

            print(f"Using fixed mu: {self.mu}")


        # dynamic viscosity
        mu = Constant(self.mu)

        # density
        rho = Constant(flow_params["rho"])

        if geometry_params["wall_jets"]:

            mesh = Mesh()
            with XDMFFile(MPI.comm_world, "mesh/two_jets_mesh.xdmf") as xdmf:
                xdmf.read(mesh)

            mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
            with XDMFFile(MPI.comm_world, "mesh/two_jets_facets.xdmf") as xdmf:
                xdmf.read(mvc, "name_to_read")

            surfaces = MeshFunction("size_t", mesh, mvc)

            inlet_tag = 1
            outlet_tag = 2
            wall_tag1 = 3
            wall_tag2 = 4
            wall_tag3 = 5
            wall_tag4 = 8
            wall_tag5 = 9
            wall_tag6 = 10
            jet1_tag = 6
            jet2_tag = 7

        else:

            mesh = Mesh()
            with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
                xdmf.read(mesh)
            # Read the mesh from the XDMF file
            # Create a MeshFunction to store the facets of the mesh

            mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
            with XDMFFile(MPI.comm_world, facet_file) as f:
                # Read the facets from the XDMF file
                f.read(mvc, "name_to_read")

            # function that extracts the entities of dimension equal to the one of the
            # topology minus one from a mesh
            surfaces = MeshFunction("size_t", mesh, mvc)

            inlet_tag = 1
            outlet_tag = 2
            wall_tag1 = 3
            wall_tag2 = 4
            wall_tag3 = 5
            jet1_tag = 6

        self.domain_area = assemble(Constant(1.0) * dx(mesh))

        # Define function spaces for velocity and pressure
        # continuous galerkin polynomials of degreee 2 and 1 respectively
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)
        # functions for explicit terms
        u_n, p_n = Function(V), Function(Q)

        # External clock
        gtime = 0.0

        if geometry_params["wall_jets"]:
            # Read the velocity and pressure data from XDMF files
            with XDMFFile("mesh/two_jets_init/u.xdmf") as velocity_xdmf:
                velocity_xdmf.read_checkpoint(u_n, "u0", 0)

            with XDMFFile("mesh/two_jets_init/p.xdmf") as pressure_xdmf:
                pressure_xdmf.read_checkpoint(p_n, "p0", 0)

        # Initialize u_n and p_n from u_init and p_init xdmf files where we have last step of baseline
        # simulation
        else:
            for path, func, name in zip(("u_init", "p_init"), (u_n, p_n), ("u0", "p0")):
                if path in flow_params:
                    comm = mesh.mpi_comm()
                    XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)
        # Functions for solution at each step
        u_, p_ = Function(V), Function(Q)

        # Temporal step
        dt = Constant(solver_params["dt"])

        # Define expressions used in variational forms

        U = Constant(0.5) * (u_n + u)

        # Normal versor exiting any cell
        n = FacetNormal(mesh)

        # Forcing homogeneous term
        f = Constant((0, 0))

        # Symmetrica part of the gradient
        epsilon = lambda u: sym(nabla_grad(u))

        # Cauchy stress tensor
        sigma = lambda u, p: 2 * mu * epsilon(u) - p * Identity(2)

        # Fractional step method Chorin Temam
        # with consistency correction for pressure

        # Define variational problem for step 1 (advection diffusion problem)
        # u solution of step 1
        F1 = (
            rho * dot((u - u_n) / dt, v) * dx  # time derivative
            + rho
            * dot(dot(u_n, nabla_grad(u_n)), v)
            * dx  # transport non linear term trated implicitely
            + inner(sigma(U, p_n), epsilon(v))
            * dx  # diffusion semi-implicit with pressure correction
            + dot(p_n * n, v) * ds  # Neumann pressure bcs
            - dot(mu * nabla_grad(U) * n, v) * ds  # Neumann velocity bcs
            - dot(f, v) * dx
        )  # forcing term

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2 (laplacian problem for pressure)
        a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / dt) * div(u_) * q * dx

        # Define variational problem for step 3 (projection step)
        a3 = dot(u, v) * dx
        # u_ here will be the solution of the step 1 and the solution of this step will
        # be stored again in u_
        L3 = dot(u_, v) * dx - dt * dot(nabla_grad(p_ - p_n), v) * dx

        # extract boundary condition at the inlet in closed form
        inflow_profile = flow_params["inflow_profile"](mesh, degree=2)

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
        length_before_control = geometry_params["length_before_control"]
        control_width = geometry_params["control_width"]
        step_height = geometry_params["step_height"]
        jet_amplitude_tuning = geometry_params["tuning_parameters"][0]
        frequency_amplitude_tuning = geometry_params["tuning_parameters"][1]
        frequency_shift_tuning = geometry_params["tuning_parameters"][2]

        control_width_wall = 0.01

        jet_tag = jet1_tag

        if geometry_params["wall_jets"]:

            bcu_wall4 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag4)
            bcu_wall5 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag5)
            bcu_wall6 = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag6)

            ymax1 = (step_height * 0.75) + control_width_wall
            ymax2 = step_height * 0.25

            self.jet1 = WallJetBCValue(
                gtime,
                True,
                ymax1,
                ymax2,
                control_width_wall,
                jet_amplitude_tuning,
                frequency_amplitude_tuning,
                frequency_shift_tuning,
                frequency=0,
                Q=0,
                degree=1,
            )
            self.jet2 = WallJetBCValue(
                gtime,
                False,
                ymax1,
                ymax2,
                control_width_wall,
                -jet_amplitude_tuning,
                frequency_amplitude_tuning,
                frequency_shift_tuning,
                frequency=0,
                Q=0,
                degree=1,
            )

            bcu_jet1 = DirichletBC(V, self.jet1, surfaces, jet1_tag)
            bcu_jet2 = DirichletBC(V, self.jet2, surfaces, jet2_tag)

            bcu = [
                bcu_inlet,
                bcu_wall1,
                bcu_wall2,
                bcu_wall3,
                bcu_wall4,
                bcu_wall5,
                bcu_wall6,
                bcu_jet1,
                bcu_jet2,
            ]

        else:

            self.jet = JetBCValue(
                gtime,
                length_before_control,
                step_height,
                control_width,
                jet_amplitude_tuning,
                frequency_amplitude_tuning,
                frequency_shift_tuning,
                frequency=0,
                Q=0,
                degree=1,
            )

            # Boundary condition for jet, here set as no-slip
            bcu_jet = DirichletBC(V, self.jet, surfaces, jet_tag)
            bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3, bcu_jet]

        # All bcs objects together (where we don't impose anything we have homogeneous Neumann)
        # pressure bcs
        bcp = [bcp_outflow]

        # Initialize matrices for algebraic Chorin Temam
        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [
            SystemAssembler(a1, L1, bcu),
            SystemAssembler(a2, L2, bcp),
            SystemAssembler(a3, L3, bcu),
        ]

        # Apply bcs to matrices
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get("solver_type")
        assert solver_type in ("lu", "la_solve")
        la_solver1 = solver_params.get("la_solver_step_1")
        la_solver2 = solver_params.get("la_solver_step_2")
        la_solver3 = solver_params.get("la_solver_step_3")
        precond1 = solver_params.get("preconditioner_step_1")
        precond2 = solver_params.get("preconditioner_step_2")
        precond3 = solver_params.get("preconditioner_step_3")

        # diret solver
        if solver_type == "lu":
            solvers = list(map(lambda x: LUSolver(), range(3)))

        # iterative solver
        else:
            # we have to afind reasonable preconditioners
            solvers = [
                KrylovSolver(la_solver1, precond1),
                KrylovSolver(la_solver2, precond2),
                KrylovSolver(la_solver3, precond3),
            ]

        # Set matrices
        for s, A in zip(solvers, As):
            s.set_operator(A)
            # set parameters for iterative method
            if not solver_type == "lu":
                s.parameters["relative_tolerance"] = 1e-8
                s.parameters["monitor_convergence"] = False

        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n = p_, p_n

        # Rename u_, p_ with standard names
        u_.rename("velocity", "0")
        p_.rename("pressure", "0")

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure("ds", domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.jet_tag = jet_tag

    def evolve(self, jet_bc_values):
        """
        Make one time step dt with the given values of jet boundary conditions
        """

        # Update jet amplitude and frequency
        self.jet.Q = jet_bc_values[0]

        if self.geometry_params["set_freq"]:
            self.jet.freq = self.geometry_params["fixed_freq"]

        else:
            self.jet.freq = jet_bc_values[1]

        # Increments time
        self.gtime += self.dt(0)

        self.jet.time = self.gtime

        # Updating inflow profile if it's function of time
        inflow = self.inflow_profile
        if hasattr(inflow, "time"):
            inflow.time = self.gtime

        # solvers from the object
        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        # solving the 3 steps respectively in u_ p_ and u_ again
        for assembler, b, solver, uh in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            solver.solve(uh.vector(), b)

        # updating last step parameters
        u_n.assign(u_)
        p_n.assign(p_)

        # return next step solution
        return u_, p_

    def two_jets_evolve(self, jet_bc_values):
        """
        Make one time step dt with the given values of jet boundary conditions
        """

        # Update jet amplitude and frequency
        self.jet1.Q = jet_bc_values[0]
        self.jet2.Q = jet_bc_values[0]

        if self.geometry_params["set_freq"]:
            self.jet1.freq = 0.01
            self.jet2.freq = 0.01

        else:
            self.jet1.freq = jet_bc_values[1]
            self.jet2.freq = jet_bc_values[1]

        # Increments time
        self.gtime += self.dt(0)

        self.jet1.time = self.gtime
        self.jet2.time = self.gtime

        # Updating inflow profile if it's function of time
        inflow = self.inflow_profile
        if hasattr(inflow, "time"):
            inflow.time = self.gtime

        # solvers from the object
        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        # solving the 3 steps respectively in u_ p_ and u_ again
        for assembler, b, solver, uh in zip(assemblers, bs, solvers, (u_, p_, u_)):
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

    bot = mesh.coordinates().min(axis=0)[1] + 0.1
    top = mesh.coordinates().max(axis=0)[1]

    # width of inlet channel
    H = top - bot

    #
    U_in = 1.5

    return Expression(
        ("-4*U_in*(x[1]-bot)*(x[1]-top)/H/H", "0"),
        bot=bot,
        top=top,
        H=H,
        U_in=U_in,
        degree=degree,
    )
