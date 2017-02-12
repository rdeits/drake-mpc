from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.spatial as spat
import matplotlib as mpl
import matplotlib.pyplot as plt
import irispy as iri
import itertools as it

def dare(A, B, Q, R):
    """
    dare(A, B, Q, R):
    returns the solution of the discrete-time algebraic Riccati equation
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    Q -> state stage weight matrix
    R -> input stage weight matrix
    OUTPUTS:
    P -> solution of the DARE
    K -> optimal gain matrix
    """
    # DARE solution 
    P = la.solve_discrete_are(A,B,Q,R)
    # optimal gain
    K = - la.inv(B.T.dot(P).dot(B)+R).dot(B.T).dot(P).dot(A)
    return [P, K]

def moas(A, lhs_x, rhs_x):
    """
    moas(A, lhs_x, rhs_x):
    returns the maximum output admissibile set (i.e., the set such that
    if x(0) \in moas then lhs_x*x(k) <= rhs_x with x(k)=A*x(k-1) \forall k>=1)
    Algorithm 3.2 from "Gix_lbert, Tan - Linear Systems with State and Control Constraints:
    The Theory and Application of Maximal Output Admissible Sets"
    INPUTS:
    A -> state transition matrix
    [lhs_x, rhs_x] -> state constraint lhs_x*x(k) <= rhs_x \forall k
    OUTPUTS:
    [lhs_moas, rhs_moas] -> matrices such that moas := {x | lhs_moas*x <= rhs_moas}
    t                  -> number of steps after that the constraints are redundant
    """
    n_x = A.shape[0]
    t = 0
    convergence = False
    while convergence == False:
        # cost function jacobians for all i
        J = lhs_x.dot(np.linalg.matrix_power(A,t+1))
        # constraints to each LP
        cons_lhs = np.vstack([lhs_x.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([rhs_x for k in range(0,t+1)])
        ## list of all minima
        s = rhs_x.shape[0]
        J_sol = [(-lin_or_quad_prog(np.array([]), -J[i,:].T, cons_lhs, cons_rhs)[1] - rhs_x[i]) for i in range(0,s)]
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1
    lhs_moas = cons_lhs
    rhs_moas = cons_rhs
    return [lhs_moas, rhs_moas, t]

def linSysEvo(A, B, N):
    """
    linSysEvo(A, B, N):
    returns the free and forced evolution matrices
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    N -> time steps
    OUTPUTS:
    [for_evo, free_evo] -> matrices such that x_traj = for_evo*u_seq + free_evo*x_0
    """
    # forced evolution of the system
    [n_x, n_u] = B.shape
    for_evo = np.zeros((n_x*N,n_u*N))
    for i in range(0, N):
        for j in range(0, i+1):
            for_evo_ij = np.linalg.matrix_power(A,i-j).dot(B)
            for_evo[n_x*i:n_x*(i+1),n_u*j:n_u*(j+1)] = for_evo_ij
    # free evolution of the system
    free_evo = np.vstack([np.linalg.matrix_power(A,k) for k in range(1, N+1)])
    return [for_evo, free_evo]

def sim_lin_sys(x0, N, A, B=0, u_seq=0):
    """
    sim_lin_sys(x0, N, A, B=0, u_seq=0):
    simulates the evolution of the linear system
    INPUTS:
    x0    -> initial conditions
    N     -> number of simulation steps
    A     -> state transition matrix
    B     -> input to state matrix
    u_seq -> input sequence \in R^(N*n_u)
    OUTPUTS:
    x_traj -> state trajectory \in R^(N*n_x)
    """
    # system dimensions
    n_x = A.shape[1]
    if np.any(B) != 0:
        n_u = B.shape[1]
    # state trajectory
    x_traj = x0
    for k in range(1, N+1):
        x_prev = x_traj[n_x*(k-1):n_x*k]
        x_next = A.dot(x_prev)
        if np.any(B) != 0:
            u = u_seq[n_u*(k-1):n_u*k]
            x_next = x_next + B.dot(u)
        x_traj = np.vstack((x_traj, x_next))
    return x_traj

def ocp_cost_fun(A, B, Q, R, P, N):
    """
    ocp_cost_fun(A, B, Q, R, P, N):
    returns the cost function blocks of the ocp QP
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    Q -> state weight matrix of the LQR problem
    R -> input weight matrix of the LQR problem
    N -> time steps
    OUTPUTS:
    H -> Hessian of the ocp QP
    F -> cost function linear term (to be left-multiplied by x_0^T!)
    """
    # quadratic term in the state sequence
    q_st = la.block_diag(*[Q for k in range(0, N-1)])
    q_st = la.block_diag(q_st,P)
    # quadratic term in the input sequence
    q_in = la.block_diag(*[R for k in range(0, N)])
    # evolution of the system
    [for_evo, free_evo] = linSysEvo(A,B,N)
    # quadratic term
    H = 2*(q_in+for_evo.T.dot(q_st.dot(for_evo)))
    # linear term
    F = 2*for_evo.T.dot(q_st.T).dot(free_evo)
    F = F.T
    return [H, F]

def ocp_cons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_xN, rhs_xN, N):
    """
    ocp_cons(lhs_u, rhs_u, lhs_x, rhs_x, lhs_xN, rhs_xN, N):
    returns the constraint blocks of the ocp QP
    INPUTS:
    A                -> state transition matrix
    B                -> input to state matrix
    [lhs_u, rhs_u]   -> input constraint lhs_u*u(k) <= rhs_u \forall k
    [lhs_x, rhs_x]   -> state constraint lhs_x*x(k) <= rhs_x \forall k
    [lhs_xf, rhs_xf] -> final state constraint lhs_xN*x(N) <= rhs_xN
    N            -> time steps
    OUTPUTS:
    [G, W, E] -> QP constraints such that G*u_seq <= W + E*x0
    """
    n_x = lhs_x.shape[1]
    # input constraints
    G_u = la.block_diag(*[lhs_u for k in range(0, N)])
    W_u = np.vstack([rhs_u for k in range(0, N)])
    E_u = np.zeros((W_u.shape[0],n_x))
    # evolution of the system
    [for_evo, free_evo] = linSysEvo(A, B, N)
    # state constraints
    lhs_x_diag = la.block_diag(*[lhs_x for k in range(1, N+1)])
    G_x = lhs_x_diag.dot(for_evo)
    W_x = np.vstack([rhs_x for k in range(1, N+1)])
    E_x = - lhs_x_diag.dot(free_evo)
    # terminal constraints
    G_xN = lhs_xN.dot(for_evo[-n_x:,:])
    W_xN = rhs_xN
    E_xN = - lhs_xN.dot(np.linalg.matrix_power(A,N))
    # gather constraints
    G = np.vstack((G_u, G_x, G_xN))
    W = np.vstack((W_u, W_x, W_xN))
    E = np.vstack((E_u, E_x, E_xN))
    return [G, W, E]

def lin_or_quad_prog(H, f, A, b, x_bound=1e6):
    """
    Solves a quaratic or a linear program, depending on the input H
    INPTS:
    [H, f]  -> cost function min .5 x' H x + f' x
    [A, b]  -> constraints A x <= b
    x_bound -> bound on the solution
    OUTPUTS:
    x_min    -> x such that the cost assumes its minimum value
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
    if H.size:
        prog.AddQuadraticCost(H, f, x)
    else:
        prog.AddLinearCost((f + 1e-15).T.dot(x))
    # set bounds to the solution
    for i in range(0, n):
            prog.AddLinearConstraint(x[i] <= x_bound)
            prog.AddLinearConstraint(x[i] >= -x_bound)
    # solve
    solver = GurobiSolver()
    result = solver.Solve(prog)
    x_min = prog.GetSolution(x).reshape(n,1)
    # look for unbounded programs
    toll = 1e-6
    if not any(np.isnan(x_min)):
        if any(np.absolute(x_min) >= x_bound - toll):
            print("Unbounded LP in the domain ||x||_inf <= " + str(x_bound))
    # retrieve solution
    if H.size:
        cost_min = .5*x_min.T.dot(H.dot(x_min)) + f.T.dot(x_min)
    else:
        cost_min = f.T.dot(x_min)
    return [x_min, cost_min]

def qp_builder(A, B, Q, R, u_min, u_max, x_min, x_max, N_ocp, ter_cons):
    """
    ocp_builder(A, B, Q, R, u_min, u_max, x_min, x_max, N_ocp):
    Computes th blocks of a QP given the blocks of a OCP
    INPTS:
    A -> state transition matrix
    B -> input to state matrix
    Q -> state stage weight matrix
    R -> input stage weight matrix
    [u_min, u_max] -> bounds on the inputs
    [x_min, x_max] -> bounds on the states
    N_ocp -> ocp horizon
    ter_cons -> terminal constraints imposed to the OCP (available options: 'moas', 'none')
    OUTPUTS:
    H -> Hessian of the ocp QP
    F -> cost function linear term (to be left-multiplied by x_0^T!)
    [G, W, E] -> QP constraints such that G*u_seq <= W + E*x0
    """
    n_x = A.shape[0]
    n_u = B.shape[1]
    # stage constraints
    lhs_u = np.vstack((np.eye(n_u), -np.eye(n_u)))
    rhs_u = np.vstack((u_max, -u_min))
    lhs_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
    rhs_x = np.vstack((x_max, -x_min))
    # terminal constraints
    if ter_cons == 'moas':
        # solve dare
        [P, K] = dare(A, B, Q, R)
        A_cl = A + B.dot(K)
        # compute moas
        lhs_x_cl = np.vstack((lhs_x,lhs_u.dot(K)))
        rhs_x_cl = np.vstack((rhs_x,rhs_u))
        [lhs_moas, rhs_moas, t] = moas(A_cl, lhs_x_cl, rhs_x_cl)
        poly_moas = Poly(lhs_moas, rhs_moas)
        lhs_xN = poly_moas.lhs
        rhs_xN = poly_moas.rhs
    # elif ter_cons == 'origin':
    #     P = np.zeros((n_x,n_x))
    #     lhs_xN = np.vstack((np.eye(n_x), - np.eye(n_x)))
    #     rhs_xN = np.zeros((2*n_x,1))
    elif ter_cons == 'none':
        P = dare(A, B, Q, R)[0]
        lhs_xN = np.array([]).reshape(0,n_x)
        rhs_xN = np.array([]).reshape(0,1)
    # compute blocks
    [G, W, E] = ocp_cons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_xN, rhs_xN, N_ocp)
    [H, F] = ocp_cost_fun(A, B, Q, R, P, N_ocp)
    # remove always-redundant constraints (coincident constraints are extremely problematic!)
    poly_cons = Poly(np.hstack((G, -E)), W)
    G = poly_cons.lhs[:,:n_u*N_ocp]
    E = - poly_cons.lhs[:,n_u*N_ocp:]
    W = poly_cons.rhs
    return [H, F, G, W, E]

class Poly:
    """
    Defines a polyhedron as {x | lhs x <= rhs}
    ATTRIBUTES:
        [lhs, rhs]         -> left- and right-hand-side of the (possibly redundant) representation of the polyhedron
        x_bound            -> bounding box dimension
        [lhs_min, rhs_min] -> left- and right-hand-side of the MINIMAL representation of the polyhedron
        verts              -> vertices of the polyhedron
    """

    def __init__(self, lhs, rhs, x_bound=1e6):
        ### scale polyhedron equations to avoid numeric errors
        for i in range(0,lhs.shape[0]):
            norm_fact = np.linalg.norm(np.hstack((lhs[i,:],rhs[i])))
            if norm_fact > 1e-3:
                lhs[i,:] = lhs[i,:]/norm_fact
                rhs[i] = rhs[i]/norm_fact

        ### ensure that the polyhedron is not empty
        x_feasible = lin_or_quad_prog(np.array([]), np.zeros(lhs[0,:].shape), lhs, rhs, x_bound)[0]
        self.x_bound = x_bound
        self.empty = any(np.isnan(x_feasible))

        ### minimal representation of the polyhedron
        n = lhs.shape[1]
        m = lhs.shape[0]
        if self.empty:
            print('Empty polyhedron!')
        else:
            self.coinc_facets = poly_coinc_facets(lhs,rhs)
            [self.lhs, self.rhs, self.min_facets_indices] = poly_min_facets(lhs, rhs, x_bound)
     
            ### vertices of the polyhedron
            # bounding box
            lhs_bound = np.vstack((self.lhs, np.eye(n), -np.eye(n)))
            rhs_bound = np.vstack((self.rhs, x_bound*np.ones((2*n,1))))
            poly = iri.Polyhedron(lhs_bound, rhs_bound)
            self.verts = np.vstack(poly.generatorPoints())
            toll = 1e-6
            if any(np.absolute(self.verts).flatten() >= x_bound - toll):
                print("Unbounded polyhedron in the domain ||x||_inf <= " + str(x_bound))

    def plot(self, line_style='b'):
        """
        Plots the "D" polyhedron in the (x_1,x_2) plane
        INPUTS:
        line_style -> specs of the plot color and line style
        OUTPUTS:
        poly_plot -> figure handle
        """
        if self.empty:
            raise ValueError('Empty polyhedron!')
        n = self.lhs.shape[1]
        if n > 2:
            raise ValueError('Unable to plot polyhedrons in more than 2 dimensions!')
        hull = spat.ConvexHull(self.verts)
        for simp in hull.simplices:
            self.poly_plot, = plt.plot(self.verts[simp, 0], self.verts[simp, 1], line_style)
        toll = 1e-6
        vert_unb = np.array([]).reshape(0,2)
        for i in range(self.verts.shape[0]):
            if any(np.absolute(self.verts[i,:]) >= self.x_bound - toll):
                vert_unb = np.vstack((vert_unb, self.verts[i,:]))
        if any(np.absolute(self.verts.flatten()) >= self.x_bound - toll):
            plt.scatter(vert_unb[:,0], vert_unb[:,1], color='r', alpha=.5, label='Unbounded vertices')
            plt.legend(loc=1)
        # ### plot center
        # center = np.mean(self.verts[:-1,:], axis=0)
        # plt.scatter(center[0], center[1], color='g')
        # ###
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        return self.poly_plot

def poly_facet_center(poly, lhs_facet, rhs_facet):
    """
    derives the center of a facet which belongs to a polyhedron
    INPUTS:
    poly -> polyhedron
    [lhs_facet, rhs_facet] -> equation of the facet (lhs_facet * x = rhs_facet)
    OUTPUTS:
    center -> coordinates of the center of the polyhedron facet
    """
    facet_edeges = np.array([]).reshape(0,len(lhs_facet))
    toll = 1e-9
    for vert in poly.verts:
        if np.absolute(lhs_facet.dot(vert.T)-rhs_facet) < toll:
            facet_edeges = np.vstack((facet_edeges, vert))
    if facet_edeges.shape[0] < 2:
        print '!!! This error can be caused by numeric issues !!!'
        print 'At least two of the following values'
        for vert in poly.verts:
            print lhs_facet.dot(vert.T)-rhs_facet
        print 'should be zero with a tollerance of ' + str(toll) + '.'
        print 'These are generated by the expression:'
        print str(lhs_facet)
        print 'times each one of the following'
        print poly.verts
        print 'minus'
        print str(rhs_facet) + '.'
        print 'This is the plot of the region ...'
        poly.plot()
        plt.show()
        raise ValueError('The given equation is not a facet of the polyhedron!')
    center = np.mean(facet_edeges, axis=0)
    return center.reshape(len(center),1)

def poly_coinc_facets(lhs,rhs):
    """
    poly_coinc_facets(lhs,rhs):
    returns the coincident factes of a polyhedron
    INPUTS:
    [lhs, rhs] -> left- and right-hand-side of the representation of the polyhedron
    OUTPUTS:
    coinc_facets -> list of lists of coincident facets
    """
    m = lhs.shape[0]
    # concatenate left- and right-hand sides
    lrhs = np.hstack((lhs,rhs))
    # normalize each row
    for i in range(0, m):
        lrhs[i,:] = lrhs[i,:]/(np.linalg.norm(lrhs[i,:]))
    # list of lists of coincident facets
    coinc_facets = []
    # look for coincident facets
    for i in range(0, m):
        # if the ith facet is not already be found to be coincident to someother facets
        if i not in [j for sublist in coinc_facets for j in sublist]:
            coinc_facets_i = [i]
            # check coincidence with all the others
            for j in range(i+1, m):
                if all(np.isclose(lrhs[i,:], lrhs[j,:])):
                    coinc_facets_i.append(j)
            # if coincidence append to coinc_facets
            coinc_facets.append(coinc_facets_i)
    return coinc_facets

def poly_min_facets(lhs, rhs, x_bound=1e6):
    """
    poly_min_facets(lhs, rhs, x_bound=1e6):
    returns the minimum set of factes to represent a polyhedron (in case of coincident facets, only one is mantained)
    INPUTS:
    [lhs, rhs] -> left- and right-hand-side of the representation of the polyhedron
    x_bound    -> bound for unbounded polyhedron
    OUTPUTS:
    [lhs_min, rhs_min] -> minimal representation of the polyhedron
    min_facets_indices -> indices of the non-redundant facets in the original representation
    """
    n = lhs.shape[1]
    m = lhs.shape[0]
    # list of non-redundant facets
    min_facets_indices = range(0, m)
    for i in range(0, m):
        # remove redundant constraints
        lhs_i = lhs[min_facets_indices,:]
        # relax the ith constraint
        rhs_relax = np.zeros(np.shape(rhs))
        rhs_relax[i] += 1
        rhs_i = (rhs + rhs_relax)[min_facets_indices];
        # check redundancy
        f_i = -lhs[i,:].T
        cost_i = lin_or_quad_prog(np.array([]), f_i, lhs_i, rhs_i, x_bound)[1]
        cost_i = - cost_i - rhs[i]
        # remove redundant facets from the list
        toll = 1e-6
        if np.absolute(cost_i) < toll:
            print "Coincident hyperplanes with tollerance " + str(cost_i[0])
        if cost_i < toll:
            min_facets_indices.remove(i)
    min_facets_indices = min_facets_indices
    lhs_min = lhs[min_facets_indices,:]
    rhs_min = rhs[min_facets_indices]
    return [lhs_min, rhs_min, min_facets_indices]

class CriticalRegion:
    # this is structured following:
    # Tondel, Johansen, Bemporad - An algorithm for multi-parametric quadratic programming and explicit MPC solutions

    def __init__(self, act_set, H, G, W, S):

        ### active set
        self.act_set = act_set
        self.inact_set = list(set(range(0, G.shape[0])) - set(act_set))
        n_inact = len(self.inact_set)

        ### optimal solution as a function of x
        H_inv = np.linalg.inv(H)
        # active and inactive constraints
        [G_A, W_A, S_A] = [G[self.act_set,:], W[self.act_set,:], S[self.act_set,:]]
        [G_I, W_I, S_I] = [G[self.inact_set,:], W[self.inact_set,:], S[self.inact_set,:]]
        # explicit solution for dual variables (lam_A_opt (x) = lam_A_trans + lam_A_lin * x)
        H_a = np.linalg.inv(G_A.dot(H_inv.dot(G_A.T))) # (always invertible since licq holds)
        self.lam_A_trans = - H_a.dot(W_A)
        self.lam_A_lin = - H_a.dot(S_A)
        # explicit solution for primal variables (z_opt (x) = z_A_trans + z_lin * x)
        self.z_trans = - H_inv.dot(G_A.T.dot(self.lam_A_trans))
        self.z_lin = - H_inv.dot(G_A.T.dot(self.lam_A_lin))

        ### state-space polyhedron 
        # equation (12) (revised, only inactive indices...)
        lhs_t1 = G_I.dot(self.z_lin) - S_I
        rhs_t1 = - G_I.dot(self.z_trans) + W_I
        # equation (13)
        lhs_t2 = H_a.dot(S_A)
        rhs_t2 = - H_a.dot(W_A)
        # construct polyhedron
        self.poly_t12 = Poly(np.vstack((lhs_t1, lhs_t2)), np.vstack((rhs_t1, rhs_t2)))
        # if the polyhedron is empty return 
        if self.poly_t12.empty:
            return

        ### candidate active sets for the neighboring critical regions
        # detect coincident facets
        coinc_facets_list = coinc_facets_types(self.act_set, self.inact_set, self.poly_t12)
        # candidate active sets (without considering weakly active constraints)
        neig_act_set_list = neig_act_set_generator(self.act_set, self.inact_set, coinc_facets_list, self.poly_t12)
        # detect weakly active constraints
        [weakly_act, weakly_act_set] = weak_act_set_detector(self.act_set, lhs_t2, rhs_t2)
        # candidate active sets (considering weakly active constraints)
        if weakly_act:
            # add all the new candidate sets to the list
            neig_act_set_list_weak = neig_act_set_if_weak(weakly_act_set, neig_act_set_list)
            neig_act_set_list += neig_act_set_list_weak
        self.neig_act_set_list = neig_act_set_list

    def z_opt(self, x):
        """
        Return the explicit solution of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        z_opt -> solution of the QP
        """
        z_opt = self.z_trans + self.z_lin.dot(x)
        return z_opt

    def lam_opt(self, x):
        """
        Return the explicit value of the multipliers of the mpQP as a function of the parameter
        INPUTS:
        x -> value of the parameter
        OUTPUTS:
        lam_opt -> optimal multipliers
        """
        lam_A_opt = self.lam_A_trans + self.lam_A_lin.dot(x)
        lam_opt = np.zeros(G.shape[0],1)
        for i in len(self.act_set):
            lam_opt[self.act_set[i],0] = lam_A_opt[i]
        return lam_opt

def coinc_facets_types(act_set, inact_set, poly_t12):
    """
    returns the list of all the coincident facets of a critical region:
    enumerated in the as in the equation G z <= W + S x ("original enumeration")
    and and labeled as type1 or type 2
    INPUTS:
    [act_set, inact_set] -> active and inactive sets of the parent critical region
    poly_t12                           -> polyhedron describing the parent critical region
    OUTPUTS:
    coinc_facets_list -> list of coincident facts indices and types
    """
    coinc_facets_list = []
    # for all the sets of coincident facets of the polyhedron
    for facets in poly_t12.coinc_facets:
        ind_list = []
        type_list = []
        # for each facet of each set
        for facet in facets:
            # if the facet is related to an inactive constraint (type 1)
            n_inact = len(inact_set)
            if facet < n_inact:
                ind_list += [inact_set[facet]]
                type_list += ['t1']
            # if the facet is related to an active constraint (type 2)
            else:
                ind_list += [act_set[facet - n_inact]]
                type_list += ['t2']
        # add info to the list
        coinc_facets_list.append([ind_list, type_list])
    return coinc_facets_list

def neig_act_set_generator(act_set, inact_set, coinc_facets_list, poly_t12):
    """
    returns the list of candidate active sets for the neighboring regions to the one considered (called "parent"),
    Theorem 2 and Corollary 1 are here applied
    INPUTS:
    [act_set, inact_set] -> active and inactive sets of the parent critical region
    coinc_facets_list                  -> list of the coincident facets of the parent state-space polyhedron
    poly_t12                           -> polyhedron describing the parent critical region
    OUTPUTS:
    neig_act_set_list -> list of candidate active sets (including the center of the facet that generated each candidate and its gradient)
    """
    neig_act_set_list = []
    # for all the facets of this critical region
    for i in range(0,len(poly_t12.min_facets_indices)):
        # store vector normal to the shared facet (needed to cope with licq fails!))
        norm_vec = poly_t12.lhs[i,:]
        norm_vec = norm_vec.reshape(len(norm_vec),1)
        # store center of the shared facet (needed to cope with licq fails!))
        center = poly_facet_center(poly_t12, poly_t12.lhs[i,:], poly_t12.rhs[i])
        # initialize the list containing the difference of active set between parent CR and child CR
        cand_act_set_diff = []
        # start with the active set of the parent CR
        cand_act_set = act_set[:]
        # path from the index "i" to the index in the "original enumeration" (called "ind")
        ind = poly_t12.min_facets_indices[i]
        n_inact = len(inact_set)
        if ind < n_inact:
            ind = inact_set[ind]
        else:
            ind = act_set[ind - n_inact]
        # generate candidate active sets for each facet
        for coinc_facets in coinc_facets_list:
            if ind in coinc_facets[0]:
                # a new constraint becomes active (or inactive) for each coincident facet
                for j in range(0,len(coinc_facets[0])):
                    if coinc_facets[1][j] == 't1':
                        cand_act_set.append(coinc_facets[0][j])
                        cand_act_set_diff.append(coinc_facets[0][j])
                    else:
                        cand_act_set.remove(coinc_facets[0][j])
                        cand_act_set_diff.append(coinc_facets[0][j])
        cand_act_set.sort()
        # store every info about each set of coincident facets of the parent CR
        neig_act_set_list.append([cand_act_set, cand_act_set_diff, center, norm_vec])
    return neig_act_set_list

def weak_act_set_detector(act_set, lhs_t2, rhs_t2, toll=1e-6):
    """
    returns the list of constraints that are weakly active in the whole critical region
    enumerated in the as in the equation G z <= W + S x ("original enumeration")
    (by convention weakly active constraints are included among the active set, so that only constraints of type 2 are anlyzed)
    INPUTS:
    act_set          -> active set of the parent critical region
    [lhs_t2, rhs_t2] -> left- and right-hand side of the constraints of type 2 of the parent CR
    toll             -> tollerance in the detection
    OUTPUTS:
    weakly_act     -> flag determining if the CR analyzed has weakly active constraints
    weakly_act_set -> list of weakly active constraints
    """
    weakly_act = False
    weakly_act_set = []
    # weakly active constraints are included in the active set
    for i in range(0, lhs_t2.shape[0]):
        # to be weakly active in the whole region they can only be in the form 0 x <= 0 (sure ???)
        if np.linalg.norm(lhs_t2[i,:]) + np.absolute(rhs_t2[i,:]) < toll:
            print('Weakly active constraint detected!')
            # act_set[i] maps to index i into the "original enumeration"
            weakly_act_set.append(act_set[i])
            weakly_act = True
    return [weakly_act, weakly_act_set]

def neig_act_set_if_weak(weakly_act_set, neig_act_set_list):
    """
    returns the additional condidate active sets that are caused by weakly active constraints (theorem 5)
    INPUTS:
    weakly_act_set    -> indices of the weakly active contraints
    neig_act_set_list -> list of candidate neighboring active sets
    OUTPUTS:
    neig_act_set_list_weak -> additional candidate active sets to be taken into
    """
    # additional candidate active sets
    neig_act_set_list_weak = []
    # for each one of the already computed candidate active sets
    for i in range(0,len(neig_act_set_list)):
        # for every possible combination of the weakly active constraints
        for n_weakly_act in range(1,len(weakly_act_set)+1):
            for comb_weakly_act in it.combinations(weakly_act_set, n_weakly_act):
                neig_act_set_list_weak_i = []
                # remove each combination from each candidate active set to create a new candidate active set
                if set(neig_act_set_list[i][0]).issuperset(comb_weakly_act):
                    # new candidate active set
                    neig_act_set_list_weak_i.append([j for j in neig_act_set_list[i][0] if j not in list(comb_weakly_act)])
                    # update differences wrt the active set of the parent CR
                    neig_act_set_list_weak_i.append(neig_act_set_list[i][1] + [j for j in list(comb_weakly_act)])
                    # copy information of the parent's facet
                    neig_act_set_list_weak_i.append(neig_act_set_list[i][2])
                    neig_act_set_list_weak_i.append(neig_act_set_list[i][3])
                # update the list of candidate active sets generated because of wekly active constraints
                neig_act_set_list_weak.append(neig_act_set_list_weak_i)
    return neig_act_set_list_weak


def act_set_if_degeneracy(parent, ind, H, G, W, S):
    """
    returns the active set in case that licq does not hold (theorem 4 and some more...)
    INPUTS:
    parent       -> citical region that has generated this degenerate active set hypothesis
    ind          -> index of this active set hypothesis in the parent's list of neighboring active sets
    [H, G, W, S] -> cost and constraint matrices of the mp-QP
    OUTPUTS:
    act_set_child -> real active set of the child critical region (= False if the region is unfeasible)
    """
    # get the info related to the creation of the degenerate active set
    [cand_act_set, cand_act_set_diff, center, norm_vec] = parent.neig_act_set_list[ind]
    toll = 1e-6
    # if more than one constraint has been activated (I think this does not apply)
    if len(cand_act_set_diff) != 1:
        print 'Cannot solve degeneracy with multiple active set changes! The solution of a QP is required...'
        dist = 1e-6
        # just sole the QP inside the new critical region to derive the active set
        x = center + dist*norm_vec
        z = lin_or_quad_prog(H, np.zeros((H.shape[0],1)), G, W+S.dot(x))[0]
        cons_val = G.dot(z) - W - S.dot(x)
        # new active set for the child
        act_set_child = [i for i in range(0,cons_val.shape[0]) if cons_val[i] > -toll]
        # convert [] to False to avoid confusion with the empty active set...
        if not act_set_child:
            act_set_child = False
    else:
        # check that this never happens
        if cand_act_set_diff[0] < 0:
            raise ValueError('LICQ must hold when a constraint is removed!')
        # compute optimal solution in the center of the shared facet
        z_center = parent.z_opt(center)
        # solve lp from theorem 4
        G_A = G[cand_act_set,:]
        n_lam = G_A.shape[0]
        cost = np.zeros(n_lam)
        cost[cand_act_set.index(cand_act_set_diff[0])] = -1.
        cons_lhs = np.vstack((G_A.T, -G_A.T, -np.eye(n_lam)))
        cons_rhs = np.vstack((-H.dot(z_center), H.dot(z_center), np.zeros((n_lam,1))))
        lam_bound = 1e6
        lam_sol = lin_or_quad_prog(np.array([]), cost, cons_lhs, cons_rhs, lam_bound)[0]
        # if the solution in unbounded the region is not feasible
        if np.max(lam_sol) > lam_bound - toll:
            act_set_child = False
        # if the solution in bounded look at the indices of the solution
        else:
            act_set_child = []
            for i in range(0,n_lam):
                if lam_sol[i,0] > toll:
                    act_set_child += [cand_act_set[i]]
    return act_set_child

def licq_check(G, act_set):
    """
    checks if licq holds
    INPUTS:
    G -> gradient of the constraints
    act_set -> active set
    OUTPUTS:
    licq -> flag, = True if licq holds, = False otherwise
    """
    G_A = G[act_set,:]
    licq = True
    max_cond = 1e9
    cond = np.linalg.cond(G_A.dot(G_A.T))
    if cond > max_cond:
        licq = False
        print 'LICQ does not hold: condition number of G*G^T = ' + str(cond)
    return licq