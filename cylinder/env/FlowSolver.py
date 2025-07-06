from dolfin import *
import numpy as np
import os
from env.control_bcs import JetBCValue


class FlowSolver(object):
    """
    We here implement the fenics based CFD simulation used as episode template for learning and to
    generate the baseline from which every episode starts
    """

    def __init__(self, flow_params, geometry_params, solver_params):

        
        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mesh_path = os.path.join(base_dir, "mesh", f"cylinder_{self.geometry_params['num_control_jets']}jets.xdmf")
        facets_path = os.path.join(base_dir, "mesh", f"cylinder_{self.geometry_params['num_control_jets']}jets_facets.xdmf")

        # dynamic viscosity
        mu = flow_params["mu"]

        # density
        rho = flow_params["rho"]
        U_in = flow_params["U_in"]  # inflow velocity
        D = geometry_params["radius"] * 2  # cylinder diameter
        U_bar = (2 * U_in) / 3
        # Reynolds number
        self.Re = int((U_bar * rho * D) / mu)  # Reynolds number

        mu = Constant(mu)  # dynamic viscosity
        rho = Constant(rho)  # density

        mesh = Mesh()
        with XDMFFile(MPI.comm_world, mesh_path) as xdmf:
            xdmf.read(mesh)

        mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
        with XDMFFile(
            MPI.comm_world, facets_path) as xdmf:
            xdmf.read(mvc, "name_to_read")

        surfaces = MeshFunction("size_t", mesh, mvc)

        self.surfaces = surfaces

        # Define tags for the different boundaries
        inlet_tag = 3
        outlet_tag = 4
        wall_tag1 = 1  # bottom
        wall_tag2 = 2  # top
        cylinder_tag = 5  # cylinder
        self.cylinder_tag = cylinder_tag

        jet_tags = range(
            self.cylinder_tag + 1,
            self.cylinder_tag + 1 + geometry_params["num_control_jets"],
        )

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

        u_init_path = os.path.join(
            base_dir, "mesh", f"init/uRe{self.Re}_init{geometry_params['num_control_jets']}.xdmf"
        )
        p_init_path = os.path.join(
            base_dir, "mesh", f"init/pRe{self.Re}_init{geometry_params['num_control_jets']}.xdmf"
        )

        # Read the velocity and pressure data from XDMF files
        with XDMFFile(u_init_path) as velocity_xdmf:
            velocity_xdmf.read_checkpoint(u_n, "velocity", 0)

        with XDMFFile(p_init_path) as pressure_xdmf:
            pressure_xdmf.read_checkpoint(p_n, "pressure", 0)

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

        if geometry_params["num_control_jets"] == 2:
            jet_angles = [90.0, 270.0]  # jet angles
        elif geometry_params["num_control_jets"] == 3:
            jet_angles = [0.0, 105.0, 255.0]  # jet angles
        elif geometry_params["num_control_jets"] == 5:
            jet_angles = [0.0, 45, 105, 255.0, 315.0]  # jet angles

        self.jet_angles_rad = [
            np.radians(angle) for angle in jet_angles
        ]  # Convert to radians
        self.center = self.geometry_params["center"]
        self.radius = self.geometry_params["radius"]
        self.width = self.geometry_params["width"]  # jet width

        bcu_jets = []

        # Create and assign JetBCValue instances as self.jet1, self.jet2, ...
        for i, theta0 in enumerate(self.jet_angles_rad):
            jet = JetBCValue(
                gtime, self.center, 0.0, theta0, self.width, self.radius, degree=2
            )
            setattr(self, f"jet{i}", jet)

        # Now access them again by name and build the BCs
        for i, tag in enumerate(jet_tags):
            jet_attr = getattr(self, f"jet{i}")  # access self.jet1, self.jet2, ...
            bc = DirichletBC(V, jet_attr, surfaces, tag)
            bcu_jets.append(bc)

        # # Define boundary conditions
        print("jet ok")
        # # Inlet velocity
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
        # # No slip
        bcu_wall1 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, wall_tag1)
        bcu_wall2 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, wall_tag2)
        bcu_wall3 = DirichletBC(V, Constant((0.0, 0.0)), surfaces, self.cylinder_tag)

        # # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0.0), surfaces, outlet_tag)

        bcu = [bcu_inlet, bcu_wall1, bcu_wall2, bcu_wall3] + bcu_jets

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

    def evolve(self, jet_bc_values):
        """
        Make one time step dt with the given values of jet boundary conditions
        """

        # # Update jet amplitude and frequency
        # self.jet1.Q = jet_bc_values[0]
        # self.jet2.Q = jet_bc_values[1]
        # self.jet3.Q = jet_bc_values[2]
        # # self.jet4.Q = jet_bc_values[3]
        # # self.jet5.Q = jet_bc_values[4]

        # # Increments time
        # self.gtime += self.dt(0)

        # self.jet1.time = self.gtime
        # self.jet2.time = self.gtime
        # self.jet3.time = self.gtime
        # # self.jet4.time = self.gtime
        # # self.jet5.time = self.gtime

        self.gtime += self.dt(0)

        for i, Q in enumerate(jet_bc_values):
            jet = getattr(self, f"jet{i}", None)
            if jet is not None:
                jet.Q = Q
                jet.time = self.gtime
            else:
                raise ValueError(f"Jet {i} not found. Did you initialize enough jets?")

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

    def compute_drag_lift(self):
        mu = float(self.viscosity)

        # Symmetrica part of the gradient
        epsilon = lambda u: sym(nabla_grad(u))

        # Cauchy stress tensor
        sigma = lambda u, p: 2 * mu * epsilon(u) - p * Identity(2)

        u = self.u_
        p = self.p_
        n = FacetNormal(p.function_space().mesh())
        ds = Measure(
            "ds", domain=p.function_space().mesh(), subdomain_data=self.surfaces
        )

        traction = dot(sigma(u, p), n)
        drag = -traction[0] * ds(
            self.cylinder_tag
        )  # Integrate over the cylinder surface
        lift = -traction[1] * ds(
            self.cylinder_tag
        )  # Integrate over the cylinder surface

        drag_force = assemble(drag)
        lift_force = assemble(lift)
        rho = float(self.density)  # Default to 1.0
        U_bar = (2 * self.flow_params["U_in"]) / 3  # Set in your inflow profile
        D = self.radius * 2  # Cylinder diameter

        Cd = 2 * drag_force / (rho * U_bar**2 * D)
        Cl = 2 * lift_force / (rho * U_bar**2 * D)
        return Cd, Cl


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
