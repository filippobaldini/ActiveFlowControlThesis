import numpy as np
from mpi4py import MPI as py_mpi
from dolfin import *

# try:
#     from iufl import icompile
#     from iufl.corealg.traversal import traverse_unique_terminals
#     from iufl.operators import eigw

# except ImportError:
#     print('iUFL can be obtained from https://github.com/MiroK/ufl-interpreter')


class Probe_Evaluator(object):
    '''Perform efficient evaluation of function (u or p) at fixed points'''
    def __init__(self, eval_func, probes_locations):
        # The idea here is that u(x) means: search for cell containing x,
        # evaluate the basis functions of that element at x, restrict
        # the coef vector of u to the cell. Of these 3 steps the first
        # two don't change. So we cache them

        # Locate each point in the grid
        mesh = eval_func.function_space().mesh()
        total_cell_number = mesh.num_entities(mesh.topology().dim())
        # Creates object bounding box tree for research cell purposes
        bbox_tree = mesh.bounding_box_tree()

        # Initialize a list of cells with the same length of probes_locations
        cells_list = [None]*len(probes_locations)

        # cycle used to examinate the entire probes_locations vector,
        # we find for each point_x in probes_locations the exact cell where it is (through bbox) and assign it to the cell list
        for i, point_x in enumerate(probes_locations):
            cell = bbox_tree.compute_first_entity_collision(Point(*point_x))
            from dolfin import info
            if -1 < cell < total_cell_number:
                cells_list[i] = cell

        func_space = eval_func.function_space()
        element = func_space.dolfin_element()

        size = func_space.ufl_element().value_size()

        # Build now the sampling matrix to evaluate the function, that is an empty vector for the values
        # and a map for the degrees of freedom (both for each cell that in general, thanks to .dofmap()
        evals = []
        dof_map = func_space.dofmap()


        for point_x, cell in zip(probes_locations, cells_list):
            # Control that we have the cell in the list, compute the basis function and then initialize an element for the coefficients to evaluate
            if cell is not None:
                basis_matrix = np.zeros(size*element.space_dimension())
                # 1 if it is pressure, 2 if it is velocity
                coefficients = np.zeros(element.space_dimension())
                # maps the local indexes to the global ones for the single cell, adding also the first global index for the current parallel environment
                celldofs = dof_map.cell_dofs(cell) + dof_map.ownership_range()[0]
                cell = Cell(mesh, cell)
                vertex_coords, orientation = cell.get_vertex_coordinates(), cell.orientation()
                # Eval the basis function for the current cell through flattening the matrix in a vector by .ravel()
                basis_matrix.ravel()[:] = element.evaluate_basis_all(point_x, vertex_coords, orientation)
                basis_matrix = basis_matrix.reshape((element.space_dimension(), size)).T
                

                def foo(function_vec, c=coefficients, A=basis_matrix, dofs=celldofs):
                    # Restrict for each call using the bound cell, vc ...
                    c[:] = function_vec.getValues(dofs)
                    return np.dot(A, c)

            # Otherwise we use the value which plays nicely with MIN reduction
            else:
                foo = lambda u, size=size: (np.finfo(float).max)*np.ones(size)

            evals.append(foo)
        self.probes = evals

        # recall parallel attributes
        self.comm = py_mpi.COMM_WORLD
        self.readings = np.zeros(size*len(probes_locations), dtype=float)
        self.readings_local = np.zeros_like(self.readings)
        # Return the value in the shape of vector/matrix
        self.nprobes = len(probes_locations)

        # this is the function called when created/evaluated the functions at the probes : return the values in the probes_locations
        # important to give the same names as in "foo"
    def sample(self, u):
        '''Evaluate the probes listing the time as t'''
        function_vec = as_backend_type(u.vector()).vec()  # This is PETSc
        self.readings_local[:] = np.hstack([f(function_vec) for f in self.probes])    # Get local
        self.comm.Reduce(self.readings_local, self.readings, op=py_mpi.MIN)  # Sync

        return self.readings.reshape((self.nprobes, -1))



# Both this and Velocity probes are called in Env2DCyl after the creation of the flow object, they are child_classes of Point_probe
class PressureProbeValues(Probe_Evaluator):
    '''Point value of pressure at locations'''
    def __init__(self, flow, locations):
        Probe_Evaluator.__init__(self, flow.p_, locations)

    def sample(self, u, p): return Probe_Evaluator.sample(self, p)





class VelocityProbeValues(Probe_Evaluator):
    '''Point value of velocity vector at locations'''
    def __init__(self, flow, locations):
        Probe_Evaluator.__init__(self, flow.u_, locations)

    def sample(self, u, p): return Probe_Evaluator.sample(self, u)



# This function is basically the same implemented by Rabault: it

class TotalrecirculationArea(object):
    '''
    Approximate recirculation area based on thresholding the horizontal component
    of the velocity within a spatial region given by a geometric predicate passed as in-line argument of the function.
    With non-empy path a MeshFunction marking the recirculation bubble
    is saved at each `sample` call.
    We used as geometric predicate a function "include_cell" which by default includes all the cells in the geometry mesh
    '''

    def __init__(self, velocity, threshold, include_cell =lambda x: True, store_path=''):
        assert velocity.function_space().ufl_element().family() == 'Lagrange'

        # Saving the mesh dofs the in a structure (self.indices)
        V = velocity.function_space()
        # keep the dof_map only for the horizontal component of the velocity
        dof_map = V.sub(0).dofmap()
        self.indices = dof_map.dofs()
        mesh = velocity.function_space().mesh()

        #insert all the cells in our case, but obviously changing the geometry predicate, we can change the selection and choose only a specific region .
        cells_selected = filter(lambda cell_: include_cell(cell_.midpoint().array()), cells(mesh))
        cells_selected, vol_cells_selected = list(zip(*((cell_.index(), cell_.volume()) for cell_ in cells_selected)))

        # We say that the cell is inside iff it is selected before
        cellsdof = [set(dof_map.cell_dofs(cell)) for cell in cells_selected]

        # Need to remember the geometry candidate : these attributes will be used in the sample() function which is
        # the method that actually calculates the recirculation area.
        self.cells_selected = cells_selected
        self.vol_cells_selected = vol_cells_selected
        self.cellsdof = cellsdof
        self.threshold = threshold
        #initialize this attribute of the object, used in sample()
        self.recirculation_cells = None

        # if store_path:
        #     out = File(store_path)
        #     f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        #     f_array = f.array()
        #     self.counter = 0

        #     def dump(self, f=f, array=f_array, out=out):
        #         if(not self.recirculation_cells is None):
        #             f.set_all(0)  # Reset
        #             array[self.recirculation_cells] = 1  # Set
        #             out << (f, float(self.counter))  # Dump
        #             self.counter += 1

        #     self.dump = dump
        # else:
        #     self.dump = lambda foo, bar: None

    # note: even if the function is basically implemented also for taking into account the pressure,
    # we decided to stick, as Rabault already did, on the calculus of the Recirculation area through the velocity
    def sample(self, velocity, pressure):
        # All point evaluation of the velocity
        all_coefs = velocity.vector().get_local()
        # restriction to the points selected before according to the geometry predicate
        dof_coefs = all_coefs[self.indices]
        # Filter among those vertices where the velocity coefficient does not reach the threshold
        coefficient_filter = np.where(dof_coefs < self.threshold)[0]
        # build an array of the filtered dofs
        filtered_dofs = set(np.array(self.indices)[coefficient_filter])  # Now global w.r.t to V
        # A cell with some dofs which are filtered, is counted as a cell where there is recirculation .
        self.recirculation_cells = [cell for cell, dofs in enumerate(self.cellsdof) if dofs & filtered_dofs]

        recirc_area = sum(self.vol_cells_selected[cell] for cell in self.recirculation_cells)

        return recirc_area



#if __name__ == '__main__':
    # from dolfin import *
    # mesh = UnitSquareMesh(64, 64)

    # #########################
    # # Check scalar
    # #########################
    # V = FunctionSpace(mesh, 'CG', 2)
    # f = Expression('t*(x[0]+x[1])', t=0, degree=1)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    # f_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    #
    # u = interpolate(f, V)
    # locations = np.array([[0.2, 0.2],
    #                       [0.8, 0.8],
    #                       [1.0, 1.0],
    #                       [0.5, 0.5]])
    #
    # probes = PointProbe(u, locations)
    #
    # for t in [0.1, 0.2, 0.3, 0.4]:
    #     f.t = t
    #     u.assign(interpolate(f, V))
    #     Sample f
    #     ans = probes.sample(u)
    #     truth = f_(t, locations).reshape((len(locations), -1))
        # NOTE: that the sample always return as matrix, in particular
        # for scale the is npoints x 1 matrix
        # assert np.linalg.norm(ans - truth) < 1E-14, (ans, truth)

    # ##########################
    # # Check vector
    # ##########################
    # V = VectorFunctionSpace(mesh, 'CG', 2)
    # f = Expression(('t*(x[0]+x[1])',
    #                 't*x[0]*x[1]'), t=0, degree=2)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    # f0_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    # f1_ = lambda t, x: t*x[:, 0]*x[:, 1]
    #
    # u = interpolate(f, V)
    # locations = np.array([[0.2, 0.2],
    #                       [0.8, 0.8],
    #                       [1.0, 1.0],
    #                       [0.5, 0.5]])
    #
    # probes = PointProbe(u, locations)
    #
    # for t in [0.1, 0.2, 0.3, 0.4]:
    #     f.t = t
    #     u.assign(interpolate(f, V))
        # Sample f
        # ans = probes.sample(u)
        # truth = np.c_[f0_(t, locations).reshape((len(locations), -1)),
        #               f1_(t, locations).reshape((len(locations), -1))]
        #
        # assert np.linalg.norm(ans - truth) < 1E-14, (ans, truth)

    ##########################
    # Check expression
    ##########################
    # V = VectorFunctionSpace(mesh, 'CG', 2)
    # f = Expression(('t*(x[0]+x[1])',
    #                 't*(2*x[0] - x[1])'), t=0.0, degree=2)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    # f0_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    # f1_ = lambda t, x: t*(2*x[:, 0]-x[:, 1])
    #
    # f_ = lambda t, x: f0_(t, x)**2 + f1_(t, x)**2
    #
    # u = interpolate(f, V)
    # locations = np.array([[0.2, 0.2],
    #                       [0.8, 0.8],
    #                       [1.0, 1.0],
    #                       [0.5, 0.5]])
    #
    # probes = ExpressionProbe(inner(u, u), locations)
    #
    # for t in [0.1, 0.2, 0.3, 0.4]:
    #     f.t = t
    #     u.assign(interpolate(f, V))
        # Sample f
        # ans = probes.sample()
        # truth = f_(t, locations).reshape((len(locations), -1))
        #
        # assert np.linalg.norm(ans - truth) < 1E-14, (t, ans, truth)

    # Now for something fancy
    # from iufl.operators import eigw
    # Eigenvalues of outer product of velocity
    # probes = ExpressionProbe(eigw(outer(u, u)), locations, mesh=mesh)
    #
    # for t in [0.1, 0.2, 0.3, 0.4]:
    #     f.t = t
    #     u.assign(interpolate(f, V))
    #     Sample f
        # ans = probes.sample()
        # print(ans)

    # Recirc area
    # mesh = UnitSquareMesh(4, 4)
    # V = VectorFunctionSpace(mesh, 'CG', 2)
    # f = Expression(('x[0]-0.5', '0'), degree=2)
    #
    # v = interpolate(f, V)  # Function like FlowSolver.u_
    #
    # probe = RecirculationAreaProbe(v,
    #                                threshold=0,
    #                            geom_predicate=lambda x: 0.25 < x[1] < 0.75,
    #                                store_path='./recirc_area.pvd')
    # print(probe.sample(v, None))
    #
    # v.assign(interpolate(Expression(('x[0]-0.75', '0'), degree=2), V))
    # print(probe.sample(v, None))
#