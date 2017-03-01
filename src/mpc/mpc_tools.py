from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np
import scipy.linalg as la
import scipy.spatial as spat
import matplotlib as mpl
import matplotlib.pyplot as plt
import irispy as iris
import itertools as it
import mpl_toolkits.mplot3d as a3
import time


def plot_input_sequence(u_sequence, t_s, N, u_max=None, u_min=None):
    """
    plots the input sequence "u_sequence" and its bounds "u_max" and "u_min" as functions of time
    INPUTS:
    t_s -> sampling time
    N -> number of steps
    u_sequence -> input sequence, list of dimension (number of steps = N)
    u_max -> upper bound on the inputs of dimension (number of inputs)
    u_min -> lower bound on the inputs of dimension (number of inputs)
    OUTPUTS:
    none
    """
    n_u = u_sequence[0].shape[0]
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        u_i_sequence = [u_sequence[j][i] for j in range(0,N)]
        input_plot, = plt.step(t, [u_i_sequence[0]] + u_i_sequence, 'b')
        if u_max is not None:
            bound_plot, = plt.step(t, u_max[i,0]*np.ones(t.shape), 'r')
        if u_min is not None:
            bound_plot, = plt.step(t, u_min[i,0]*np.ones(t.shape), 'r')
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if u_max is not None or u_min is not None:
                plt.legend([input_plot, bound_plot], ['Optimal control', 'Control bounds'], loc=1)
            else:
                plt.legend([input_plot], ['Optimal control'], loc=1)
    plt.xlabel(r'$t$')

def plot_state_trajectory(x_trajectory, t_s, N, x_max=None, x_min=None):
    """
    plots the state trajectory "x_trajectory" and its bounds "x_max" and "x_min" as functions of time
    INPUTS:
    x_trajectory -> state trajectory, list of dimension (number of steps + 1= N + 1)
    t_s -> sampling time
    N -> number of steps
    x_max -> upper bound on the states of dimension (number of states)
    x_min -> lower bound on the states of dimension (number of states)
    OUTPUTS:
    none
    """
    n_x = x_trajectory[0].shape[0]
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        x_i_trajectory = [x_trajectory[j][i] for j in range(0,N+1)]
        state_plot, = plt.plot(t, x_i_trajectory, 'b')
        if x_max is not None:
            bound_plot, = plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        if x_min is not None:
            bound_plot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            if x_max is not None or x_min is not None:
                plt.legend([state_plot, bound_plot], ['Optimal trajectory', 'State bounds'], loc=1)
            else:
                plt.legend([state_plot], ['Optimal trajectory'], loc=1)
    plt.xlabel(r'$t$')

def linear_program(f, A, b, x_bound=1e8):
    """
    solves the linear program min f^T * x s.t. { A * x <= b , ||x||_inf <= x_bound }
    INPUTS:
    f -> gradient of the cost function
    A -> left hand side of the constraints
    b -> right hand side of the constraints
    x_bound -> bound on the infinity norm of the solution
    OUTPUTS:
    x_min -> argument which minimizes
    cost_min -> minimum of the cost function
    """
    # program dimensions
    n = f.shape[0]
    m = A.shape[0]
    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    for i in range(0, m):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddLinearCost((f + 1e-15).T.dot(x))
    # set bounds to the solution
    for i in range(0, n):
            prog.AddLinearConstraint(x[i] <= x_bound)
            prog.AddLinearConstraint(x[i] >= -x_bound)
    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = prog.GetSolution(x).reshape(n,1)
    # retrieve solution
    cost_min = f.T.dot(x_min)
    return [x_min, cost_min]

def quadratic_program(H, f, A, b):
    """
    solves the quadratic program min x^t * H * x + f^T * x s.t. A * x <= b
    INPUTS:
    H -> Hessian of the cost function
    f -> linear term of the cost function
    A -> left hand side of the constraints
    b -> right hand side of the constraints
    OUTPUTS:
    x_min -> argument which minimizes
    cost_min -> minimum of the cost function
    """
    # program dimensions
    n = f.shape[0]
    m = A.shape[0]
    # build program
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    for i in range(0, m):
        prog.AddLinearConstraint((A[i,:] + 1e-15).dot(x) <= b[i])
    prog.AddQuadraticCost(H, f, x)
    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = prog.GetSolution(x).reshape(n,1)
    cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)
    return [x_min, cost_min]

class Polyhedron:

    def __init__(self, lhs, rhs):
        """
        defines a polyhedron lhs * x <= rhs
        """
        # halfplanes
        self.lhs = lhs
        self.rhs = rhs
        # size
        [self.n_facets, self.n_vars] = lhs.shape
        # store some data
        self.normalize()
        self.is_empty()
        self.coincident_facets()
        self.minimal_facets()
        self.vertices()
        self.facet_centers()

    def normalize(self, toll=1e-6):
        """
        normalizes each equation of the polyhedron with the norm of the left hand side
        INPUTS:
        toll -> minimum norm of the left hand side under which the normalization is not performed
        """
        for i in range(0, self.n_facets):
            norm_factor = np.linalg.norm(self.lhs[i,:])
            if norm_factor > toll:
                self.lhs[i,:] = self.lhs[i,:]/norm_factor
                self.rhs[i] = self.rhs[i]/norm_factor
        return

    def is_empty(self, x_bound=1e8):
        """
        checks if the polyhedron is empty
        INPUTS:
        x_bound -> bound on the infinity norm of the points inside the polyhedron
        """
        x_feasible = linear_program(np.zeros(self.n_vars), self.lhs, self.rhs, x_bound)[0]
        self.empty = any(np.isnan(x_feasible))
        return

    def coincident_facets(self, toll=1e-8):
        """
        for each facet it lists the coincident facets
        INPUTS:
        toll -> tollerance in the detection of coincident facets
        """
        # coincident facets indices
        coincident_facets = []
        lrhs = np.hstack((self.lhs, self.rhs))
        for i in range(0, self.n_facets):
            coincident_facets.append(
                np.where(
                    np.all(
                        np.isclose(lrhs, lrhs[i,:], toll, toll),axis=1))[0].tolist())
        self.coincident_facets = coincident_facets
        return

    def minimal_facets(self, toll=1e-8):
        """
        finds the non-redundant facets and derives a minimal representation of the polyhedron
        INPUTS:
        toll -> tollerance in the detection of redundant facets
        """
        # list of non-redundant facets
        minimal_facets = range(0, self.lhs.shape[0])
        for i in range(0, self.lhs.shape[0]):
            # remove redundant constraints
            lhs_i = self.lhs[minimal_facets,:]
            # relax the ith constraint
            rhs_relaxation = np.zeros(np.shape(self.rhs))
            rhs_relaxation[i] += 1
            rhs_relaxed = (self.rhs + rhs_relaxation)[minimal_facets];
            # check redundancy
            cost_i = linear_program(-self.lhs[i,:].T, lhs_i, rhs_relaxed)[1]
            cost_i = - cost_i - self.rhs[i]
            # remove redundant facets from the list
            if cost_i < toll:
                minimal_facets.remove(i)
        # list of non redundant facets
        self.minimal_facets = minimal_facets
        self.lhs_min = self.lhs[self.minimal_facets,:]
        self.rhs_min = self.rhs[self.minimal_facets]
        return

    def vertices(self, toll=1e-3, x_bound=1e8):
        """
        computes the vertices of the polyhedron
        INPUTS:
        toll -> tollerance in the detection of unboundedness
        x_bound -> bound on the infinity norm of the polyhedron (used to check unboundedness)
        """
        lhs_bounded = np.vstack((self.lhs_min, np.eye(self.n_vars), -np.eye(self.n_vars)))
        rhs_bounded = np.vstack((self.rhs_min, x_bound*np.ones((2*self.n_vars,1))))
        p = iris.Polyhedron(lhs_bounded, rhs_bounded)
        self.vertices = p.generatorPoints()
        if any(np.absolute(np.vstack(self.vertices)).flatten() >= x_bound - toll):
            print("Warning: unbounded polyhedron in the domain ||x||_inf <= " + str(x_bound))
        return
    
    def facet_centers(self, toll=1e-8):
        """
        computes the center of each non-reundant facet of the polyhedron
        INPUTS:
        toll -> tollerance in the coupling of facets with vertices
        """
        # first derive the vertices of the facets
        facet_vertices = []
        for i in range(0, len(self.minimal_facets)):
            vertices_fac_i = []
            for vert in self.vertices:
                if np.absolute(self.lhs_min[i,:].dot(vert)-self.rhs_min[i]) < toll:
                    vertices_fac_i.append(vert)
            if len(vertices_fac_i) < self.n_vars:
                print '(This error is likely to be caused by numeric issues ...)'
                raise ValueError('The given equation is not a facet of the polyhedron!')
            facet_vertices.append(vertices_fac_i)
        # now find the centers
        facet_centers = []
        for vertices in facet_vertices:
            facet_centers.append(np.mean(np.vstack(vertices), axis=0))
        self.facet_centers = facet_centers

    def plot(self, dim_proj=[0,1], **kwargs):
        """
        plots a 2d projection of the polyhedron
        INPUTS:
        line_style -> line style
        dim_proj -> dimensions in which to project the polyhedron
        OUTPUTS:
        polyhedron_plot -> figure handle
        """
        if self.empty:
            raise ValueError('Empty polyhedron!')
        if len(dim_proj) != 2:
            raise ValueError('Only 2d polyhedrons!')
        # extract vertices components
        vertices_proj = np.vstack(self.vertices)[:,dim_proj]
        hull = spat.ConvexHull(vertices_proj)
        for simplex in hull.simplices:
            polyhedron_plot, = plt.plot(vertices_proj[simplex, 0], vertices_proj[simplex, 1], **kwargs)
        plt.xlabel(r'$x_' + str(dim_proj[0]+1) + '$')
        plt.ylabel(r'$x_' + str(dim_proj[1]+1) + '$')
        return polyhedron_plot

    # def plot3d(self, dim_proj=[0,1,2], **kwargs):
    #     """
    #     plots a 3d projection of the polyhedron
    #     INPUTS:
    #     line_style -> line style
    #     dim_proj -> dimensions in which to project the polyhedron
    #     OUTPUTS:
    #     polyhedron_plot -> figure handle
    #     """
    #     if self.empty:
    #         raise ValueError('Empty polyhedron!')
    #     if len(dim_proj) != 3:
    #         raise ValueError('Only 3d polyhedrons!')
    #     # extract vertices components
    #     vertices_proj = np.vstack(self.vertices)[:,dim_proj]
    #     hull = spat.ConvexHull(vertices_proj)
    #     ax = a3.Axes3D(plt.gcf())
    #     for simplex in hull.simplices:
    #         poly = a3.art3d.Poly3DCollection([vertices_proj[simplex]], **kwargs)
    #         poly.set_edgecolor('k')
    #         ax.add_collection3d(poly)
    #     plt.xlabel(r'$x_' + str(dim_proj[0]+1) + '$')
    #     plt.ylabel(r'$x_' + str(dim_proj[1]+1) + '$')
    #     plt.zlabel(r'$x_' + str(dim_proj[2]+1) + '$')
    #     return

    @staticmethod
    def from_bounds(x_max, x_min):
        """
        defines a polyhedron from a set of bounds
        INPUTS:
        x_max -> upper bound
        x_min -> lower bound
        OUTPUTS:
        p -> polyhedron
        """
        n = x_max.shape[0]
        lhs = np.vstack((np.eye(n), -np.eye(n)))
        rhs = np.vstack((x_max, -x_min))
        p = Polyhedron(lhs, rhs)
        return p

class DTLinearSystem:

    def __init__(self, A, B=None):
        self.A = A
        self.n_x = np.shape(A)[0]
        if B is None:
            B = np.array([]).reshape(self.n_x, 0)
        self.B = B
        self.n_u = np.shape(B)[1]
        self.state_domain = None
        self.input_domain = None
        return

    def add_state_domain(self, lhs, rhs):
        if self.state_domain is not None:
            print('Warning: overwriting state domain!')
        self.state_domain = Polyhedron(lhs, rhs)
        return

    def add_state_bounds(self, x_max, x_min):
        if self.state_domain is not None:
            print('Warning: overwriting state domain!')
        self.state_domain = Polyhedron.from_bounds(x_max, x_min)

    def add_input_domain(self, lhs, rhs):
        if self.input_domain is not None:
            print('Warning: overwriting input domain!')
        self.input_domain = Polyhedron(lhs, rhs)
        return

    def add_input_bounds(self, u_max, u_min):
        if self.input_domain is not None:
            print('Warning: overwriting input domain!')
        self.input_domain = Polyhedron.from_bounds(u_max, u_min)

    def evolution_matrices(self, N):
        # free evolution of the system
        free_evolution = np.vstack([np.linalg.matrix_power(self.A,k) for k in range(1, N+1)])
        # forced evolution of the system
        forced_evolution = np.zeros((self.n_x*N,self.n_u*N))
        for i in range(0, N):
            for j in range(0, i+1):
                forced_evolution[self.n_x*i:self.n_x*(i+1),self.n_u*j:self.n_u*(j+1)] = np.linalg.matrix_power(self.A,i-j).dot(self.B)
        return [free_evolution, forced_evolution]

    def simulate(self, x0, N, u_sequence=None):
        if u_sequence is None:
            u_sequence = np.array([]).reshape(self.n_u*N, 0)
        [free_evolution, forced_evolution] = self.evolution_matrices(N)
        # state trajectory
        x_trajectory = free_evolution.dot(x0) + forced_evolution.dot(u_sequence)
        x_trajectory = np.vstack((x0, x_trajectory))
        return x_trajectory

    def maximum_output_admissible_set(self):
        if np.max(np.absolute(np.linalg.eig(self.A)[0])) > 1:
            raise ValueError('Cannot compute MOAS for unstable systems')
        if self.n_u > 0:
            print('Warning: computing MOAS for actuated system')
        t = 0
        convergence = False
        while convergence == False:
            # cost function jacobians for all i
            J = self.state_domain.lhs_min.dot(np.linalg.matrix_power(self.A,t+1))
            # constraints to each LP
            cons_lhs = np.vstack([self.state_domain.lhs_min.dot(np.linalg.matrix_power(self.A,k)) for k in range(0,t+1)])
            cons_rhs = np.vstack([self.state_domain.rhs_min for k in range(0,t+1)])
            # list of all minima
            s = len(self.state_domain.minimal_facets)
            J_sol = [(-linear_program(-J[i,:].T, cons_lhs, cons_rhs)[1] - self.state_domain.rhs_min[i]) for i in range(0,s)]
            if np.max(J_sol) < 0:
                convergence = True
            else:
                t += 1
        # remove redundant constraints
        moas = Polyhedron(cons_lhs, cons_rhs)
        return [moas, t]

    @staticmethod
    def from_continuous(t_s, A, B=None):
        n_x = np.shape(A)[0]
        if B is None:
            B = np.array([]).reshape(n_x, 0)
        n_u = np.shape(B)[1]
        mat_c = np.zeros((n_x+n_u, n_x+n_u))
        mat_c[0:n_x,:] = np.hstack((A,B))
        mat_d = la.expm(mat_c*t_s)
        A_d = mat_d[0:n_x, 0:n_x]
        B_d = mat_d[0:n_x, n_x:n_x+n_u]
        sys = DTLinearSystem(A_d, B_d)
        return sys

class MPCController:

    def __init__(self, sys, Q, R, N, terminal_cost=None, terminal_constraint=None):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.N = N
        self.terminal_cost = terminal_cost
        self.terminal_cost_matrix()
        self.terminal_constraint = terminal_constraint
        self.constraint_blocks()
        self.cost_blocks()
        self.critical_regions = None

    def dare(self):
        # DARE solution
        P = la.solve_discrete_are(self.sys.A, self.sys.B, self.Q, self.R)
        # optimal gain
        K = - la.inv(self.sys.B.T.dot(P).dot(self.sys.B)+self.R).dot(self.sys.B.T).dot(P).dot(self.sys.A)
        return [P, K]

    def terminal_constraint_polyhedron(self):
        if self.terminal_constraint is None:
            lhs_xN = np.array([]).reshape(0,self.sys.n_x)
            rhs_xN = np.array([]).reshape(0,1)
        elif self.terminal_constraint == 'moas':
            # solve dare
            K = self.dare()[1]
            # closed loop dynamics
            A_cl = self.sys.A + self.sys.B.dot(K)
            sys_cl = DTLinearSystem(A_cl)
            # constraints for the maximum output admissible set
            state_domain_cl_lhs = np.vstack((self.sys.state_domain.lhs_min, self.sys.input_domain.lhs_min.dot(K)))
            state_domain_cl_rhs = np.vstack((self.sys.state_domain.rhs_min, self.sys.input_domain.rhs_min))
            sys_cl.add_state_domain(state_domain_cl_lhs, state_domain_cl_rhs)
            # compute maximum output admissible set
            moas = sys_cl.maximum_output_admissible_set()[0]
            lhs_xN = moas.lhs_min
            rhs_xN = moas.rhs_min
        elif self.terminal_constraint == 'origin':
            lhs_xN = np.vstack((np.eye(self.sys.n_x), - np.eye(self.sys.n_x)))
            rhs_xN = np.zeros((2*self.sys.n_x,1))
        else:
            raise ValueError('Unknown terminal constraint!')
        return[lhs_xN, rhs_xN]

    def constraint_blocks(self):
        # input constraints
        G_u = la.block_diag(*[self.sys.input_domain.lhs_min for i in range(0, self.N)])
        W_u = np.vstack([self.sys.input_domain.rhs_min for i in range(0, self.N)])
        E_u = np.zeros((W_u.shape[0],self.sys.n_x))
        # state constraints
        [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
        lhs_x_diag = la.block_diag(*[self.sys.state_domain.lhs_min for i in range(0, self.N)])
        G_x = lhs_x_diag.dot(forced_evolution)
        W_x = np.vstack([self.sys.state_domain.rhs_min for i in range(0, self.N)])
        E_x = - lhs_x_diag.dot(free_evolution)
        # terminal constraints
        [lhs_xN, rhs_xN] = self.terminal_constraint_polyhedron()
        G_xN = lhs_xN.dot(forced_evolution[-self.sys.n_x:,:])
        W_xN = rhs_xN
        E_xN = - lhs_xN.dot(np.linalg.matrix_power(self.sys.A, self.N))
        # gather constraints
        G = np.vstack((G_u, G_x, G_xN))
        W = np.vstack((W_u, W_x, W_xN))
        E = np.vstack((E_u, E_x, E_xN))
        # remove always-redundant constraints (coincident constraints are extremely problematic!)
        poly_cons = Polyhedron(np.hstack((G, -E)), W)
        self.G = poly_cons.lhs_min[:,:self.sys.n_u*self.N]
        self.E = - poly_cons.lhs_min[:,self.sys.n_u*self.N:]
        self.W = poly_cons.rhs_min
        return

    def terminal_cost_matrix(self):
        if self.terminal_cost is None:
            self.P = self.Q
        elif self.terminal_cost == 'dare':
            self.P = self.dare()[0]
        else:
            raise ValueError('Unknown terminalS cost!')
        return

    def cost_blocks(self):
        # quadratic term in the state sequence
        H_x = la.block_diag(*[self.Q for i in range(0, self.N-1)])
        H_x = la.block_diag(H_x, self.P)
        # quadratic term in the input sequence
        H_u = la.block_diag(*[self.R for i in range(0, self.N)])
        # evolution of the system
        [free_evolution, forced_evolution] = self.sys.evolution_matrices(self.N)
        # quadratic term
        self.H = 2*(H_u+forced_evolution.T.dot(H_x.dot(forced_evolution)))
        # linear term
        F = 2*forced_evolution.T.dot(H_x.T).dot(free_evolution)
        self.F = F.T
        return

    def feedforward(self, x0):
        u_feedforward = quadratic_program(self.H, (x0.T.dot(self.F)).T, self.G, self.W + self.E.dot(x0))[0]
        return u_feedforward

    def feedback(self, x0):
        u_feedback = self.feedforward(x0)[0:self.sys.n_u]
        return u_feedback

    def compute_explicit_solution(self):
        tic = time.clock()
        # change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
        H_inv = np.linalg.inv(self.H)
        self.S = self.E + self.G.dot(H_inv.dot(self.F.T))
        # start from the origin
        active_set = []
        cr0 = CriticalRegion(active_set, self.H, self.G, self.W, self.S)
        cr_to_be_explored = [cr0]
        explored_cr = []
        tested_active_sets =[cr0.active_set]
        # explore the state space
        while cr_to_be_explored:
            # choose the first candidate in the list and remove it
            cr = cr_to_be_explored[0]
            cr_to_be_explored = cr_to_be_explored[1:]
            if not cr.polyhedron.empty:
                # explore CR
                explored_cr.append(cr)
                # for all the facets of the CR
                for facet_index in range(0, len(cr.polyhedron.minimal_facets)):
                    # for all the candidate active sets across each facet
                    for active_set in cr.candidate_active_sets[facet_index]:
                        if active_set not in tested_active_sets:
                            tested_active_sets.append(active_set)
                            # check LICQ for the given active set
                            licq_flag = licq_check(self.G, active_set)
                            # if LICQ holds  
                            if licq_flag:
                                cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                            # correct active set if LICQ doesn't hold 
                            else:
                                print('LICQ does not hold for the active set ' + str(active_set))
                                active_set = active_set_if_not_licq(active_set, facet_index, cr, self.H, self.G, self.W, self.S)
                                if active_set:
                                    print('    corrected active set ' + str(active_set))
                                    cr_to_be_explored.append(CriticalRegion(active_set, self.H, self.G, self.W, self.S))
                                else:
                                    print('    unfeasible region detected')
        self.critical_regions = explored_cr
        toc = time.clock()
        print('\nExplicit solution successfully computed in ' + str(toc-tic) + ' s:')
        print('parameter space partitioned in ' + str(len(self.critical_regions)) + ' critical regions.')

    def evaluate_explicit_solution(self, x):
        if self.critical_regions is None:
            raise ValueError('Explicit solution not computed yet! First run .compute_explicit_solution() ...')
        # find the CR to which the test point belongs
        for cr in self.critical_regions:
            if np.max(cr.polyhedron.lhs_min.dot(x) - cr.polyhedron.rhs_min) <= 0:
                break
        # derive explicit solution
        z = cr.z_optimal(x)
        u = z - np.linalg.inv(self.H).dot(self.F.T.dot(x))
        lam = cr.lambda_optimal(x)
        cost_to_go = u.T.dot(self.H.dot(u)) + x.T.dot(self.F).dot(u)
        return [u, cost_to_go, lam] 

class CriticalRegion:
    # this is from:
    # Tondel, Johansen, Bemporad - An algorithm for multi-parametric quadratic programming and explicit MPC solutions

    def __init__(self, active_set, H, G, W, S):
        print 'Computing critical region for the active set ' + str(active_set)
        self.active_set = active_set
        self.inactive_set = list(set(range(0, G.shape[0])) - set(active_set))
        self.boundaries(H, G, W, S)
        # if the critical region is empty return 
        if self.polyhedron.empty:
            return
        self.candidate_active_sets()

    def boundaries(self, H, G, W, S):
        # optimal solution as a function of x
        H_inv = np.linalg.inv(H)
        # active and inactive constraints
        G_A = G[self.active_set,:]
        W_A = W[self.active_set,:]
        S_A = S[self.active_set,:]
        G_I = G[self.inactive_set,:]
        W_I = W[self.inactive_set,:]
        S_I = S[self.inactive_set,:]
        # multipliers explicit solution
        H_A = np.linalg.inv(G_A.dot(H_inv.dot(G_A.T)))
        self.lambda_A_constant = - H_A.dot(W_A)
        self.lambda_A_linear = - H_A.dot(S_A)
        # primal variable explicit solution 
        self.z_constant = - H_inv.dot(G_A.T.dot(self.lambda_A_constant))
        self.z_linear = - H_inv.dot(G_A.T.dot(self.lambda_A_linear))
        # equation (12) (revised, only inactive indices...)
        lhs_type_1 = G_I.dot(self.z_linear) - S_I
        rhs_type_1 = - G_I.dot(self.z_constant) + W_I
        # equation (13)
        lhs_type_2 = - self.lambda_A_linear
        rhs_type_2 = self.lambda_A_constant
        # gather facets of type 1 and 2 to define the polyhedron
        lhs = np.array([]).reshape((0,S.shape[1]))
        rhs = np.array([]).reshape((0,1))
        # gather the equations such that the ith facet is the one generated by the ith constraint
        for i in range(G.shape[0]):
            if i in self.active_set:
                lhs = np.vstack((lhs, lhs_type_2[self.active_set.index(i),:]))
                rhs = np.vstack((rhs, rhs_type_2[self.active_set.index(i),0]))
            elif i in self.inactive_set:
                lhs = np.vstack((lhs, lhs_type_1[self.inactive_set.index(i),:]))
                rhs = np.vstack((rhs, rhs_type_1[self.inactive_set.index(i),0]))
        # construct polyhedron
        self.polyhedron = Polyhedron(lhs, rhs)
        return

    def candidate_active_sets(self):
        # without considering weakly active constraints
        candidate_active_sets = candidate_active_sets_generator(self.active_set, self.polyhedron)
        # detect weakly active constraints
        weakly_active_constraints = detect_weakly_active_constraints(self.active_set, - self.lambda_A_linear, self.lambda_A_constant)
        # correct if any weakly active constraint has been detected
        if weakly_active_constraints:
            # add all the new candidate sets to the list
            candidate_active_sets = expand_candidate_active_sets(weakly_active_constraints, candidate_active_sets)
        self.candidate_active_sets = candidate_active_sets
        return

    def z_optimal(self, x):
        """
        Return the explicit solution of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        z_optimal -> solution of the QP
        """
        z_optimal = self.z_constant + self.z_linear.dot(x).reshape(self.z_constant.shape)
        return z_optimal

    def lambda_optimal(self, x):
        """
        Return the explicit value of the multipliers of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        lambda_optimal -> optimal multipliers
        """
        lambda_A_optimal = self.lambda_A_constant + self.lambda_A_linear.dot(x)
        lambda_optimal = np.zeros(len(self.active_set + self.inactive_set))
        for i in range(0, len(self.active_set)):
            lambda_optimal[self.active_set[i]] = lambda_A_optimal[i]
        return lambda_optimal

def candidate_active_sets_generator(active_set, polyhedron):
    """
    returns a condidate active set for each facet of a critical region
    Theorem 2 and Corollary 1 are here applied
    INPUTS:
    active_set  -> active set of the parent CR
    polyhedron -> polyhedron describing the parent CR
    OUTPUTS:
    candidate_active_sets -> list of candidate active sets (ordered as the facets of the parent polyhedron, i.e. lhs_min)
    """
    candidate_active_sets = []
    # for each facet of the polyhedron
    for facet in polyhedron.minimal_facets:
        # start with the active set of the parent CR
        candidate_active_set = active_set[:]
        # check if this facet has coincident facets (this list includes the facet itself)
        coincident_facets = polyhedron.coincident_facets[facet]
        # for each coincident facet
        for facet in coincident_facets:
            if facet in candidate_active_set:
                candidate_active_set.remove(facet)
            else:
                candidate_active_set.append(facet)
            candidate_active_set.sort()
            candidate_active_sets.append([candidate_active_set])
    return candidate_active_sets

def detect_weakly_active_constraints(active_set, lhs_type_2, rhs_type_2, toll=1e-8):
    """
    returns the list of constraints that are weakly active in the whole critical region
    enumerated in the as in the equation G z <= W + S x ("original enumeration")
    (by convention weakly active constraints are included among the active set,
    so that only constraints of type 2 are anlyzed)
    INPUTS:
    active_set          -> active set of the parent critical region
    [lhs_type_2, rhs_type_2] -> left- and right-hand side of the constraints of type 2 of the parent CR
    toll             -> tollerance in the detection
    OUTPUTS:
    weakly_active_constraints -> list of weakly active constraints
    """
    weakly_active_constraints = []
    # weakly active constraints are included in the active set
    for i in range(0, len(active_set)):
        # to be weakly active in the whole region they can only be in the form 0^T x <= 0
        if np.linalg.norm(lhs_type_2[i,:]) + np.absolute(rhs_type_2[i,:]) < toll:
            print('Weakly active constraint detected!')
            weakly_active_constraints.append(active_set[i])
    return weakly_active_constraints

def expand_candidate_active_sets(weakly_active_constraints, candidate_active_sets):
    """
    returns the additional condidate active sets that are caused by weakly active constraints (theorem 5)
    INPUTS:
    weakly_active_constraints    -> indices of the weakly active contraints
    candidate_active_sets -> list of candidate neighboring active sets
    OUTPUTS:
    candidate_active_sets -> complete list of candidate active sets
    """
    for i in range(0,len(candidate_active_sets)):
        # for every possible combination of the weakly active constraints
        for n_weakly_act in range(1,len(weakly_active_constraints)+1):
            for comb_weakly_act in it.combinations(weakly_active_constraints, n_weakly_act):
                candidate_active_sets_weak_i = []
                # remove each combination from each candidate active set to create a new candidate active set
                if set(candidate_active_sets[i][0]).issuperset(comb_weakly_act):
                    # new candidate active set
                    candidate_active_sets_weak_i.append([j for j in candidate_active_sets[i][0] if j not in list(comb_weakly_act)])
                # update the list of candidate active sets generated because of wekly active constraints
                candidate_active_sets[i].append(candidate_active_sets_weak_i)
    return candidate_active_sets

def active_set_if_not_licq(candidate_active_set, ind, parent, H, G, W, S, dist=1e-5, lambda_bound=1e6, toll=1e-6):
    """
    returns the active set in case that licq does not hold (theorem 4 and some more...)
    INPUTS:
    parent       -> citical region that has generated this degenerate active set hypothesis
    ind          -> index of this active set hypothesis in the parent's list of neighboring active sets
    [H, G, W, S] -> cost and constraint matrices of the mp-QP
    OUTPUTS:
    active_set_child -> real active set of the child critical region (= False if the region is unfeasible)
    """
    x_center = parent.polyhedron.facet_centers[ind]
    active_set_change = list(set(parent.active_set).symmetric_difference(set(candidate_active_set)))
    if len(active_set_change) > 1:
        print 'Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...'
        # just sole the QP inside the new critical region to derive the active set
        x = x_center + dist*parent.polyhedron.lhs_min[ind,:]
        x = x.reshape(x_center.shape[0],1)
        z = quadratic_program(H, np.zeros((H.shape[0],1)), G, W+S.dot(x))[0]
        cons_val = G.dot(z) - W - S.dot(x)
        # new active set for the child
        active_set_child = [i for i in range(0,cons_val.shape[0]) if cons_val[i] > -toll]
        # convert [] to False to avoid confusion with the empty active set...
        if not active_set_child:
            active_set_child = False
    else:
        # compute optimal solution in the center of the shared facet
        z_center = parent.z_optimal(x_center)
        # solve lp from theorem 4
        G_A = G[candidate_active_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros(n_lam)
        cost[candidate_active_set.index(active_set_change[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))

        cons_rhs = np.vstack((-H.dot(z_center), H.dot(z_center), np.zeros((n_lam,1))))
        lambda_sol = linear_program(cost, cons_lhs, cons_rhs, lambda_bound)[0]
        # if the solution in unbounded the region is not feasible
        if np.max(lambda_sol) > lambda_bound - toll:
            active_set_child = False
        # if the solution in bounded look at the indices of the solution
        else:
            active_set_child = []
            for i in range(0,n_lam):
                if lambda_sol[i,0] > toll:
                    active_set_child += [candidate_active_set[i]]
    return active_set_child

def licq_check(G, active_set, max_cond=1e9):
    """
    checks if licq holds
    INPUTS:
    G -> gradient of the constraints
    active_set -> active set
    OUTPUTS:
    licq -> flag, = True if licq holds, = False otherwise
    """
    G_A = G[active_set,:]
    licq = True
    cond = np.linalg.cond(G_A.dot(G_A.T))
    if cond > max_cond:
        licq = False
    return licq




def plot3d(poly, dim_proj=[0,1,2], **kwargs):
    """
    plots a 3d projection of the polyhedron
    INPUTS:
    line_style -> line style
    dim_proj -> dimensions in which to project the polyhedron
    OUTPUTS:
    polyhedron_plot -> figure handle
    """
    if poly.empty:
        raise ValueError('Empty polyhedron!')
    if len(dim_proj) != 3:
        raise ValueError('Only 3d polyhedrons!')
    # extract vertices coponents
    vertices_proj = np.vstack(poly.vertices)[:,dim_proj]
    hull = spat.ConvexHull(vertices_proj)
    ax = a3.Axes3D(plt.gcf())
    for simplex in hull.simplices:
        polyhedron_plot = a3.art3d.Poly3DCollection([vertices_proj[simplex]], **kwargs)
        polyhedron_plot.set_edgecolor('k')
        ax.add_collection3d(polyhedron_plot)

    return
