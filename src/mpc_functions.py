from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.spatial as spat
import matplotlib as mpl
import matplotlib.pyplot as plt
import irispy

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

def minPolyFacets(lhs, rhs):
    """
    minPolyFacets(lhs, rhs):
    removes all the redundant constraints from a polytope
    INPUTS:
    [lhs, rhs] -> polytope definition {x | lhs*x <= rhs}
    OUTPUTS:
    [lhs_min, rhs_min] -> minimal polytope definition
    min_ind            -> indices of the non-redundant constraints
    """
    n = lhs.shape[1]
    m = rhs.shape[0]
    #####
    #####
    ##### check for zero rows in the lhs
    zero_pos = []
    for i in range(0, m):
        lhs_i_norm = np.linalg.norm(lhs[i,:])
        rhs_i_norm = np.linalg.norm(rhs[i,0])
        if lhs_i_norm < 1e-9:
            if rhs_i_norm > 1e-6:
                return [np.array([]).reshape(0,n), np.array([]).reshape(0,1)]
            else:
                zero_pos.append(i)
    lhs = np.delete(lhs, zero_pos, 0)
    rhs = np.delete(rhs, zero_pos, 0)
    # look for identical constraints
    equal_pos = []
    for i in range(0, m):
        poly_i = np.hstack((lhs[i,:], rhs[i,0]))
        poly_i_unit = poly_i/(np.linalg.norm(poly_i))
        for j in range(i+1, m):
            poly_j = np.hstack((lhs[j,:], rhs[j,0]))
            poly_j_unit = poly_j/(np.linalg.norm(poly_j))
            if np.array_equal(poly_i_unit, poly_j_unit):
                equal_pos.append(i)
                break
    lhs = np.delete(lhs, equal_pos, 0)
    rhs = np.delete(rhs, equal_pos, 0)
    m = rhs.shape[0]
    #######
    #######
    #######
    # fix big bounds to the solution
    x_b = 1e6*np.ones((n,1))
    # initialize minimum size polytope
    lhs_min = np.array([]).reshape(0,n)
    rhs_min = np.array([]).reshape(0,1)
    for i in range(0, m):
        # remove the ith constraint
        lhs_i = np.delete(lhs, i, 0);
        rhs_i = np.delete(rhs, i, 0);
        # check redundancy
        prog = mp.MathematicalProgram()
        x = prog.NewContinuousVariables(n, "x")
        for j in range(0, n):
            prog.AddLinearConstraint(x[j] <= x_b[j])
            prog.AddLinearConstraint(x[j] >= -x_b[j])
        for j in range(0, m-1):
            prog.AddLinearConstraint(lhs_i[j,:].dot(x) <= rhs_i[j])
        cost_i = -lhs[i,:] + 1e-15 ### drake bug ???
        prog.AddLinearCost(cost_i.dot(x))
        solver = GurobiSolver()
        result = solver.Solve(prog)
        cost = lhs[i,:].dot(prog.GetSolution(x)) - rhs[i]
        tollerance = 1e-6
        if cost > tollerance:
            # gather non-redundant constraints
            lhs_min = np.vstack((lhs_min, lhs[i,:].reshape(1,n)))
            rhs_min = np.vstack((rhs_min, rhs[i]))
    return [lhs_min, rhs_min]

def minPolyFacetsWithType(lhs_t1, lhs_t2, rhs_t1, rhs_t2):
    """
    minPolyFacets(lhs_t1, lhs_t2, rhs_t1, rhs_t2):
    removes all the redundant facets from a polytope whose facets belong to two different groups
    INPUTS:
    [lhs_tk, rhs_tk] -> kth set of facets {x | lhs_tk * x <= rhs_tk}
    OUTPUTS:
    [lhs_min_typek, rhs_min_typek] -> non-redundant facets in the kth set
    """
    # initialize output
    n_x = lhs_t1.shape[1]
    lhs_min_type1 = np.array([]).reshape(0,n_x)
    lhs_min_type2 = np.array([]).reshape(0,n_x)
    rhs_min_type1 = np.array([]).reshape(0,1)
    rhs_min_type2 = np.array([]).reshape(0,1)
    # stack matrices of the two types
    lhs = np.vstack((lhs_t1, lhs_t2))
    rhs = np.vstack((rhs_t1, rhs_t2))
    # remove redundant facets
    [lhs_min, rhs_min] = minPolyFacets(lhs, rhs)
    # check each facet's set
    for i in range(0, rhs_min.shape[0]):
        lhs_row = lhs_min[i,:]
        rhs_row = rhs_min[i,:]
        lhs_row_list = lhs_row.tolist()
        lhs_t1_list = lhs_t1.tolist()
        # if it's type1
        if lhs_row_list in lhs_t1_list:
            if rhs_t1[lhs_t1_list.index(lhs_row_list),0] == rhs_row:
                lhs_min_type1 = np.vstack((lhs_min_type1, lhs_row))
                rhs_min_type1 = np.vstack((rhs_min_type1, rhs_row))
        # if it's type2
        else:
            lhs_min_type2 = np.vstack((lhs_min_type2, lhs_row))
            rhs_min_type2 = np.vstack((rhs_min_type2, rhs_row))
    return [lhs_min_type1, lhs_min_type2, rhs_min_type1, rhs_min_type2]

def findPolyIndex(lhs_row, rhs_row, lhs, rhs):
    """
    findPolyIndex(lhs_row, rhs_row, lhs, rhs):
    returns the row of (lhs, rhs) -> which coincides with (lhs_row,rhs_row)
    INPUTS:
    [lhs, rhs]         -> left- and right-hand-side of a set of linear equations
    [lhs_row, rhs_row] -> left- and right-hand-side of a linear equations
    OUTPUTS:
    ind -> row index
    """
    poly_row = np.hstack((lhs_row, rhs_row))
    poly = np.hstack((lhs, rhs))
    ind = np.where(np.all(poly == poly_row, axis=1))[0]
    ind = ind.tolist()[0]
    return ind


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
        # set bounds to all variables to ensure the finiteness of the solution
        x_ub = 1e6*np.ones((n_x,1))
        x_lb = -x_ub
        # cost function jacobians for all i
        J = lhs_x.dot(np.linalg.matrix_power(A,t+1))
        # constraints to each LP
        cons_lhs = np.vstack([lhs_x.dot(np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([rhs_x for k in range(0,t+1)])
        # number of state constraints
        s = rhs_x.shape[0]
        # vector of all minima
        J_sol = np.zeros(s)
        for i in range(0,s):
            # assemble LP
            prog = mp.MathematicalProgram()
            x = prog.NewContinuousVariables(n_x, "x")
            # impose constraints
            for j in range(0, cons_rhs.shape[0]):
                prog.AddLinearConstraint(cons_lhs[j,:].dot(x) <= cons_rhs[j])
            # impose bounds
            for j in range(0,n_x):
                prog.AddLinearConstraint(x[j] <= x_ub[j])
                prog.AddLinearConstraint(x[j] >= x_lb[j])
            # cost function
            J_i = -J[i,:] + 1e-15 # ???
            prog.AddLinearCost(J_i.dot(x))
            # solve LP
            solver = GurobiSolver()
            result = solver.Solve(prog)
            J_sol[i] = J[i,:].dot(prog.GetSolution(x)) - rhs_x[i]
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

def simLinSys(x0, N, A, B=0, u_seq=0):
    """
    simLinSys(x0, N, A, B=0, u_seq=0):
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

def ocpCostFun(A, B, Q, R, P, N):
    """
    ocpCostFun(A, B, Q, R, P, N):
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

def ocpCons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_xN, rhs_xN, N):
    """
    ocpCons(lhs_u, rhs_u, lhs_x, rhs_x, lhs_xN, rhs_xN, N):
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

def solveOcp(H, F, G, W, E, x0):
    """
    solveOcp(H, F, G, W, E, x0):
    returns the optimal input sequence for the givien state x0
    INPUTS:
    H         -> Hessian of the ocp QP
    F         -> cost function linear term (to be left-multiplied by x0^T!)
    [G, W, E] -> QP constraints such that G*u_seq <= W + E*x0
    x0        -> initial condition
    OUTPUTS:
    u_seq_opt -> optimal feed-forward control sequence
    """
    prog = mp.MathematicalProgram()
    n_u_seq = np.shape(G)[1]
    u_seq = prog.NewContinuousVariables(n_u_seq, "u_seq")
    prog.AddQuadraticCost(H, x0.T.dot(F).T, u_seq)
    W_x0 = W + E.dot(x0)
    for i in range(0, np.shape(W_x0)[0]):
        prog.AddLinearConstraint(G[i,:].dot(u_seq) <= W_x0[i])
    solver = GurobiSolver()
    result = solver.Solve(prog)
    print result
    u_seq_opt = prog.GetSolution(u_seq).reshape(n_u_seq,1)
    return u_seq_opt

############################################
### plot functions #########################
############################################

def plotInputSeq(u_seq, u_min, u_max, t_s, N):
    """
    plotInputSeq(u_seq, u_min, u_max, t_s, N):
    plots the input sequences as functions of time
    INPUTS:
    u_seq -> input sequence \in R^(N*n_u)
    [u_min, u_max]  -> lower and upper bound on the input
    t_s   -> sampling time
    N     -> time steps
    """
    n_u = u_seq.shape[0]/N
    u_seq = np.reshape(u_seq,(n_u,N), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        inPlot, = plt.step(t, np.hstack((u_seq[i,0],u_seq[i,:])),'b')
        x_lbPlot, = plt.step(t, u_min[i,0]*np.ones(t.shape),'r')
        plt.step(t, u_max[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend(
                [inPlot, x_lbPlot],
                ['Optimal control', 'Control bounds'],
                loc=1)
    plt.xlabel(r'$t$')

def plotStateTraj(x_traj, x_min, x_max, t_s, N):
    """
    plotStateTraj(x_traj, x_min, x_max, t_s, N):
    plots the state trajectories as functions of time
    INPUTS:
    x_traj          -> state trajectory \in R^((N+1)*n_x)
    [x_min, x_max]  -> lower and upper bound on the state
    t_s             -> sampling time
    N               -> time steps
    """
    n_x = x_traj.shape[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        stPlot, = plt.plot(t, x_traj[i,:],'b')
        x_lbPlot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')
        plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend(
                [stPlot, x_lbPlot],
                ['Optimal trajectory', 'State bounds'],
                loc=1)
    plt.xlabel(r'$t$')

def plotStateSpaceTraj(x_traj, N, col='b'):
    """
    plotStateSpaceTraj(x_traj, N, col='b'):
    plots the state trajectories as functions of time (ONLY 2D)
    INPUTS:
    x_traj -> state trajectory \in R^((N+1)*n_x)
    N      -> time steps
    col    -> line specs
    OUTPUTS:
    traj_plot -> figure handle
    """
    n_x = x_traj.shape[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    plt.scatter(x_traj[0,0], x_traj[1,0], color=col, alpha=.5)
    plt.scatter(x_traj[0,-1], x_traj[1,-1], color=col, marker='s', alpha=.5)
    traj_plot = plt.plot(x_traj[0,:], x_traj[1,:], color=col)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    return traj_plot

def plotPoly(lhs, rhs, col):
    """
    plotPoly(lhs, rhs, col):
    plots a polytope (ONLY 2D)
    INPUTS:
    [lhs, rhs] -> polytope definition {x | lhs*x <= rhs}
    col    -> line specs
    OUTPUTS:
    poly_plot -> figure handle
    """
    poly = irispy.Polyhedron(lhs, rhs)
    verts = np.vstack(poly.generatorPoints())
    verts = np.vstack((verts,verts[-1,:]))
    hull = spat.ConvexHull(verts)
    for simp in hull.simplices:
        poly_plot, = plt.plot(verts[simp, 0], verts[simp, 1], col)
    return poly_plot

def plotConsMoas(A, lhs_x, rhs_x, t):
    """
    plotConsMoas(A, lhs_x, rhs_x, t):
    plots the set of constraints that define the MOAS until the first redundant polytope
    INPUTS:
    A          -> state transition matrix
    [lhs_x, rhs_x] -> state constraint lhs_x x(k) <= rhs_x \forall k
    t          -> number of steps after that the constraints are redundant
    OUTPUTS:
    [act_plot, red_plot] -> figure handles
    """
    # plot constraints from the first redundant one
    for i in range(t+1,-1,-1):
        lhs_x_i = lhs_x.dot(np.linalg.matrix_power(A,i))
        if i == t+1:
            red_plot = plotPoly(lhs_x_i, rhs_x, 'g-.')
        else:
            act_plot = plotPoly(lhs_x_i, rhs_x, 'y-.')
    return [act_plot, red_plot]

def plotMoas(lhs_moas, rhs_moas, t, A, N=0):
    """
    plotMoas(lhs_moas, rhs_moas, t, A, N):
    plots the maximum output admissible set and a trajectory for each vertex of the moas
    INPUTS:
    [lhs_moas, rhs_moas] -> matrices such that moas := {x | lhs_moas*x <= rhs_moas}
    t                  -> number of steps after that the constraints are redundant
    A                  -> state transition matrix
    N                  -> number of steps for the simulations
    OUTPUTS:
    [moas_plot, traj_plot] -> figure handles
    """
    n_x = A.shape[0]
    # plot MOAS polyhedron
    moas_plot = plotPoly(lhs_moas, rhs_moas, 'r-')
    # simulate a trajectory for each vertex
    poly = irispy.Polyhedron(lhs_moas, rhs_moas)
    verts = np.vstack(poly.generatorPoints())
    for i in range(0, verts.shape[0]):
        vert = verts[i,:].reshape(n_x,1)
        x_traj = simLinSys(vert, N, A)
        traj_plot, = plotStateSpaceTraj(x_traj,N)
    return [moas_plot, traj_plot]

def plotNomTraj(x_traj_qp, x_traj_lqr, lhs_moas, rhs_moas):
    """
    plotNomTraj(x_traj_qp, lhs_moas, rhs_moas):
    plots the open-loop optimal trajectories for each sampling time (ONLY 2D)
    INPUTS:
    x_traj_qp            -> matrix with optimal trajectories
    x_traj_lqr           -> matrix with optimal states
    [lhs_moas, rhs_moas] -> matrices such that moas := {x | lhs_moas*x <= rhs_moas}
    """
    n_traj = np.shape(x_traj_qp)[1]
    N_ocp = (np.shape(x_traj_qp)[0]-1)/2
    col_map = plt.get_cmap('jet')
    col_norm  = mpl.colors.Normalize(vmin=0, vmax=n_traj)
    scalar_map = mpl.cm.ScalarMappable(norm=col_norm, cmap=col_map)
    poly_plot = plotPoly(lhs_moas, rhs_moas, 'r-')
    leg_plot = [poly_plot]
    leg_lab = ['MOAS']
    for i in range(0,n_traj):
        col = scalar_map.to_rgba(i)
        leg_plot += plotStateSpaceTraj(x_traj_qp[:,i], N_ocp, col)
        leg_lab += [r'$\mathbf{x}^*(x_{' + str(i) + '})$']
    for i in range(0,np.shape(x_traj_lqr)[1]):
        if i == 0:
            leg_plot += [plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)]
            leg_lab += [r'LQR']
        else:
            plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)
    plt.legend(leg_plot, leg_lab, loc=1)

class CriticalRegion:

    def __init__(self, act_set, H, G, W, S):

        ### active and inactive set
        self.act_set = act_set
        self.inact_set = list(set(range(0, G.shape[0])) - set(act_set))

        ### boundaries
        #print np.linalg.det(H)
        H_inv = np.linalg.inv(H)
        # active constraints
        G_a = G[self.act_set,:]
        W_a = W[self.act_set,:]
        S_a = S[self.act_set,:]
        # polyhedron facets divided by type
        #print np.linalg.det(G_a.dot(H_inv.dot(G_a.T)))
        H_a = np.linalg.inv(G_a.dot(H_inv.dot(G_a.T)))
        lhs_t1 = G.dot(H_inv.dot(G_a.T.dot(H_a.dot(S_a)))) - S
        rhs_t1 = - G.dot(H_inv.dot(G_a.T.dot(H_a.dot(W_a)))) + W
        lhs_t2 = H_a.dot(S_a)
        rhs_t2 = - H_a.dot(W_a)
        #######
        #######
        #######
        ### remove zeros
        lhs_t1[np.absolute(lhs_t1) < 1e-9] = 0
        lhs_t2[np.absolute(lhs_t2) < 1e-9] = 0
        rhs_t1[np.absolute(rhs_t1) < 1e-9] = 0
        rhs_t2[np.absolute(rhs_t2) < 1e-9] = 0
        # print(np.vstack((lhs_t1, lhs_t2)))
        # print(np.vstack((rhs_t1, rhs_t2)))
        ###
        #######
        #######
        #######
        # remove redundant facets
        [lhs_t1_min, lhs_t2_min, rhs_t1_min, rhs_t2_min] = minPolyFacetsWithType(lhs_t1, lhs_t2, rhs_t1, rhs_t2)
        # boundaries
        self.lhs = np.vstack((lhs_t1_min, lhs_t2_min))
        self.rhs = np.vstack((rhs_t1_min, rhs_t2_min))

        ### active sets of the neighboring critical regions
        neig_act_set = []
        act_set_copy = self.act_set
        inact_set_copy = self.inact_set
        for i in range(0, lhs_t1_min.shape[0]):
            ind = findPolyIndex(lhs_t1_min[i,:], rhs_t1_min[i,0], lhs_t1, rhs_t1)
            cons_num = inact_set_copy[ind]
            neig_i_act_set = act_set_copy + [cons_num]
            neig_act_set.append(neig_i_act_set)
        for i in range(0, lhs_t2_min.shape[0]):
            ind = findPolyIndex(lhs_t2_min[i,:], rhs_t2_min[i,0], lhs_t2, rhs_t2)
            cons_num = act_set_copy[ind]
            neig_i_act_set = act_set_copy.remove(cons_num)
            neig_act_set.append([neig_i_act_set])
        self.neig_act_set = neig_act_set