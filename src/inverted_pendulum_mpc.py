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
    K = -np.dot(np.dot(np.dot(la.inv(np.dot(np.dot(B.T,P),B)+R),B.T),P),A)
    return [P, K]

def minPolyFacets(A, b):
    """
    minPolyFacets(A, b):
    removes all the redundant constraints from a polytope
    INPUTS:
    [A, b] -> polytope definition {x | A*x <= b}
    OUTPUTS:
    [A_min, b_min] -> minimal polytope definition
    """
    # bound the problem
    n = np.shape(A)[1]
    x_ub = 1e6*np.ones((n,1))
    x_lb = -x_ub
    # initialize minimum size polytope
    A_min = np.array([]).reshape(0,n)
    b_min = np.array([]).reshape(0,1)
    for i in range(0, np.shape(b)[0]):
        # remove the ith constraint
        A_i = np.delete(A, i, 0);
        b_i = np.delete(b, i, 0);
        # check redundancy
        prog = mp.MathematicalProgram()
        x = prog.NewContinuousVariables(n, "x")
        for j in range(0, n):
            prog.AddLinearConstraint(x[j] <= x_ub[j])
            prog.AddLinearConstraint(x[j] >= x_lb[j])
        for j in range(0, np.shape(b_i)[0]):
            prog.AddLinearConstraint(np.dot(A_i[j,:], x) <= b_i[j])
        A_i = -A[i,:] + 1e-15 ### drake bug ???
        prog.AddLinearCost(np.dot(A_i,x))
        solver = GurobiSolver()
        result = solver.Solve(prog)
        cost = np.dot(A[i,:],prog.GetSolution(x)) - b[i]
        if cost > 0:
            # gather non-redundant constraints
            A_min = np.vstack((A_min, A[i,:].reshape(1,n)))
            b_min = np.vstack((b_min, b[i]))
    return [A_min, b_min]

def moas(A, A_x, b_x):
    """
    moas(A, A_x, b_x):
    returns the maximum output admissibile set (i.e., the set such that
    if x(0) \in moas then A_x*x(k) <= b_x with x(k)=A*x(k-1) \forall k>=1)
    Algorithm 3.2 from "Gix_lbert, Tan - Linear Systems with State and Control Constraints:
    The Theory and Application of Maximal Output Admissible Sets"
    INPUTS:
    A -> state transition matrix
    [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
    OUTPUTS:
    [moas_lhs, moas_rhs] -> matrices such that moas := {x | moas_lhs*x <= moas_rhs}
    t                  -> number of steps after that the constraints are redundant
    """
    n_x = np.shape(A)[0]
    t = 0
    convergence = False
    while convergence == False:
        # set bounds to all variables to ensure the finiteness of the solution
        x_ub = 1e6*np.ones((n_x,1))
        x_lb = -x_ub
        # cost function jacobians for all i
        J = np.dot(A_x, np.linalg.matrix_power(A,t+1))
        # constraints to each LP
        cons_lhs = np.vstack([np.dot(A_x, np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        cons_rhs = np.vstack([b_x for k in range(0,t+1)])
        # number of state constraints
        s = np.shape(b_x)[0]
        # vector of all minima
        J_sol = np.zeros(s)
        for i in range(0,s):
            # assemble LP
            prog = mp.MathematicalProgram()
            x = prog.NewContinuousVariables(n_x, "x")
            # impose constraints
            for j in range(0, np.shape(cons_rhs)[0]):
                prog.AddLinearConstraint(np.dot(cons_lhs[j,:], x) <= cons_rhs[j])
            # impose bounds
            for j in range(0,n_x):
                prog.AddLinearConstraint(x[j] <= x_ub[j])
                prog.AddLinearConstraint(x[j] >= x_lb[j])
            # cost function
            J_i = -J[i,:] + 1e-15 # ???
            prog.AddLinearCost(np.dot(J_i,x))
            # solve LP
            solver = GurobiSolver()
            result = solver.Solve(prog)
            J_sol[i] = np.dot(J[i,:],prog.GetSolution(x)) - b_x[i]
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1
    moas_lhs = cons_lhs
    moas_rhs = cons_rhs
    return [moas_lhs, moas_rhs, t]

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
    [n_x, n_u] = np.shape(B)
    for_evo = np.zeros((n_x*N,n_u*N))
    for i in range(0, N):
        for j in range(0, i+1):
            for_evo_ij = np.dot(np.linalg.matrix_power(A,i-j),B)
            for_evo[n_x*i:n_x*(i+1),n_u*j:n_u*(j+1)] = for_evo_ij
    # free evolution of the system
    free_evo = np.vstack([np.linalg.matrix_power(A,k) for k in range(1, N+1)])
    return [for_evo, free_evo]

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
    F -> cost function linear term (to be right-multiplied by x_0!)
    """
    # quadratic term in the state sequence
    q_st = la.block_diag(*[Q for k in range(0, N-1)])
    q_st = la.block_diag(q_st,P)
    # quadratic term in the input sequence
    q_in = la.block_diag(*[R for k in range(0, N)])
    # evolution of the system
    [for_evo, free_evo] = linSysEvo(A,B,N)
    # quadratic term
    H = 2*(q_in+np.dot(for_evo.T,np.dot(q_st,for_evo)))
    # linear term
    F = 2*np.dot(np.dot(for_evo.T,q_st.T),free_evo)
    return [H, F]

def ocpCons(A_u, b_u, A_x, b_x, A_xN, b_xN, N):
    """
    ocpCons(A_u, b_u, A_x, b_x, A_xN, b_xN, N):
    returns the constraint blocks of the ocp QP
    INPUTS:
    [A_u, b_u]   -> input constraint A_u u(k) <= b_u \forall k
    [A_x, b_x]   -> state constraint A_x x(k) <= b_x \forall k
    [A_xf, b_xf] -> final state constraint A_xN x(N) <= b_xN
    N            -> time steps
    OUTPUTS:
    [cons_lhs, cons_rhs, cons_rhs_x0] -> QP constraints such that cons_lhs*u_seq <= cons_rhs + cons_rhs_x0*x0
    """
    n_x = np.shape(A_x)[1]
    # input constraints
    in_lhs = la.block_diag(*[A_u for k in range(0, N)])
    in_rhs = np.vstack([b_u for k in range(0, N)])
    in_rhs_x0 = np.zeros((np.shape(in_rhs)[0],n_x))
    # evolution of the system
    [for_evo, free_evo] = linSysEvo(A,B,N)
    # state constraints
    diagA_x = la.block_diag(*[A_x for k in range(1, N+1)])
    st_lhs = np.dot(diagA_x, for_evo)
    st_rhs = np.vstack([b_x for k in range(1, N+1)])
    st_rhs_x0 = - np.dot(diagA_x, free_evo)
    # terminal constraints
    for_evoTer = for_evo[-n_x:,:]
    ter_lhs = np.dot(A_xN,for_evoTer)
    ter_rhs = b_xN
    ter_rhs_x0 = - np.dot(A_xN,np.linalg.matrix_power(A,N))
    # gather constraints
    cons_lhs = np.vstack((in_lhs, st_lhs, ter_lhs))
    cons_rhs = np.vstack((in_rhs, st_rhs, ter_rhs))
    cons_rhs_x0 = np.vstack((in_rhs_x0, st_rhs_x0, ter_rhs_x0))
    return [cons_lhs, cons_rhs, cons_rhs_x0]

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
    n_x = np.shape(A)[1]
    if np.any(B) != 0:
        n_u = np.shape(B)[1]
    # state trajectory
    x_traj = x0
    for k in range(1, N+1):
        x_prev = x_traj[n_x*(k-1):n_x*k]
        x_next = np.dot(A,x_prev)
        if np.any(B) != 0:
            u = u_seq[n_u*(k-1):n_u*k]
            x_next = x_next + np.dot(B,u)
        x_traj = np.vstack((x_traj, x_next))
    return x_traj

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
    n_u = np.shape(u_seq)[0]/N
    u_seq = np.reshape(u_seq,(n_u,N), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        inPlot, = plt.step(t, np.hstack((u_seq[i,0],u_seq[i,:])),'b')
        x_lbPlot, = plt.step(t, u_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, u_max[i,0]*np.ones(np.shape(t)),'r')
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
    n_x = np.shape(x_traj)[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        stPlot, = plt.plot(t, x_traj[i,:],'b')
        x_lbPlot, = plt.step(t, x_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, x_max[i,0]*np.ones(np.shape(t)),'r')
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
    n_x = np.shape(x_traj)[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    plt.scatter(x_traj[0,0], x_traj[1,0], color=col, alpha=.5)
    plt.scatter(x_traj[0,-1], x_traj[1,-1], color=col, marker='s', alpha=.5)
    traj_plot = plt.plot(x_traj[0,:], x_traj[1,:], color=col)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    return traj_plot

def plotPoly(A, b, col):
    """
    plotPoly(A, b, col):
    plots a polytope (ONLY 2D)
    INPUTS:
    [A, b] -> polytope A*x <= b
    col    -> line specs
    OUTPUTS:
    poly_plot -> figure handle
    """
    poly = irispy.Polyhedron(A, b)
    verts = np.vstack(poly.generatorPoints())
    verts = np.vstack((verts,verts[-1,:]))
    hull = spat.ConvexHull(verts)
    for simp in hull.simplices:
        poly_plot, = plt.plot(verts[simp, 0], verts[simp, 1], col)
    return poly_plot

def plotConsMoas(A, A_x, b_x, t):
    """
    plotConsMoas(A, A_x, b_x, t):
    plots the set of constraints that define the MOAS until the first redundant polytope
    INPUTS:
    A          -> state transition matrix
    [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
    t          -> number of steps after that the constraints are redundant
    OUTPUTS:
    [act_plot, red_plot] -> figure handles
    """
    # plot constraints from the first redundant one
    for i in range(t+1,-1,-1):
        A_x_i = np.dot(A_x, np.linalg.matrix_power(A,i))
        if i == t+1:
            red_plot = plotPoly(A_x_i, b_x, 'g-.')
        else:
            act_plot = plotPoly(A_x_i, b_x, 'y-.')
    return [act_plot, red_plot]

def plotMoas(moas_lhs, moas_rhs, t, A, N):
    """
    plotMoas(moas_lhs, moas_rhs, t, A, N):
    plots the maximum output admissible set and a trajectory for each vertex of the moas
    INPUTS:
    [moas_lhs, moas_rhs] -> matrices such that moas := {x | moas_lhs*x <= moas_rhs}
    t                  -> number of steps after that the constraints are redundant
    A                  -> state transition matrix
    N                  -> number of steps for the simulations
    OUTPUTS:
    [moas_plot, traj_plot] -> figure handles
    """
    # plot MOAS polyhedron
    moas_plot = plotPoly(moas_lhs, moas_rhs, 'r-')
    # simulate a trajectory for each vertex
    poly = irispy.Polyhedron(moas_lhs, moas_rhs)
    verts = np.vstack(poly.generatorPoints())
    for i in range(0, np.shape(verts)[0]):
        vert = np.reshape(verts[i,:],(n_x,1))
        x_traj = simLinSys(vert, N, A)
        traj_plot, = plotStateSpaceTraj(x_traj,N)
    return [moas_plot, traj_plot]

def plotNomTraj(x_traj_qp, x_traj_lqr, moas_lhs, moas_rhs):
    """
    plotNomTraj(x_traj_qp, moas_lhs, moas_rhs):
    plots the open-loop optimal trajectories for each sampling time (ONLY 2D)
    INPUTS:
    x_traj_qp            -> matrix with optimal trajectories
    x_traj_lqr           -> matrix with optimal states
    [moas_lhs, moas_rhs] -> matrices such that moas := {x | moas_lhs*x <= moas_rhs}
    """
    n_traj = np.shape(x_traj_qp)[1]
    N_ocp = (np.shape(x_traj_qp)[0]-1)/2
    col_map = plt.get_cmap('jet')
    col_norm  = mpl.colors.Normalize(vmin=0, vmax=n_traj)
    scalar_map = mpl.cm.ScalarMappable(norm=col_norm, cmap=col_map)
    poly_plot = plotPoly(moas_lhs, moas_rhs, 'r-')
    leg_plot = [poly_plot]
    leg_lab = ['MOAS']
    for i in range(0,n_traj-1):
        col = scalar_map.to_rgba(i)
        leg_plot += plotStateSpaceTraj(x_traj_qp[:,i], N_ocp, col)
        leg_lab += [r'$\mathbf{x}^*(x(t=' + str(i*t_s) + ' \mathrm{s}))$']
    for i in range(0,np.shape(x_traj_lqr)[1]):
        if i == 0:
            leg_plot += [plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)]
            leg_lab += [r'LQR controller']
        else:
            plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)
    plt.legend(leg_plot, leg_lab, loc=1)

def solveOcp(H, F, cons_lhs, cons_rhs, cons_rhs_x0, x):
    """
    solveOcp(H, F, cons_lhs, cons_rhs, cons_rhs_x0, x):
    returns the optimal input sequence for the givien state x
    INPUTS:
    H                              -> Hessian of the ocp QP
    F                              -> cost function linear term (to be right-multiplied by x_0!)
    [cons_lhs, cons_rhs, cons_rhs_x0] -> QP constraints such that cons_lhs*u_seq <= cons_rhs + cons_rhs_x0*x0
    x                              -> initial condition
    OUTPUTS:
    u_seq_opt -> optimal feed-forward control sequence
    """
    prog = mp.MathematicalProgram()
    n_u_seq = np.shape(cons_lhs)[1]
    u_seq = prog.NewContinuousVariables(n_u_seq, "u_seq")
    prog.AddQuadraticCost(H, np.dot(F,x), u_seq)
    cons_rhs_qp = cons_rhs + np.dot(cons_rhs_x0,x)
    for i in range(0, np.shape(cons_rhs_qp)[0]):
        prog.AddLinearConstraint(np.dot(cons_lhs[i,:], u_seq) <= cons_rhs_qp[i])
    solver = GurobiSolver()
    result = solver.Solve(prog)
    print result
    u_seq_opt = np.reshape(prog.GetSolution(u_seq),(n_u_seq,1))
    return u_seq_opt

# dynamic parameters
m = 1.
l = 1.
g = 10.

# discretization
t_s = .1
N_ocp = 3
N_mpc = 50

# system dynamics
A = np.array([[0, 1], [g/l, 0]])
B = np.array([[0], [1/(m*l**2)]])
[n_x, n_u] = np.shape(B)
C = np.eye(n_x)
n_y = np.shape(C)[0]
D = np.zeros((n_y,n_u))
sys = sig.cont2discrete((A,B,C,D),t_s,'zoh')
[A, B, C, D] = sys[0:4]

# initial state
x0 = np.array([[1.],[0.]])

# OCP cost function
Q = np.eye(n_x)/100.
R = np.eye(n_u)

# OCP constraints
u_max = np.array([[15.]])
u_min = -u_max
x_max = np.array([[1.],[1.]])
x_min = -x_max
A_u = np.vstack((np.eye(n_u), -np.eye(n_u)))
A_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
b_u = np.vstack((u_max, -u_min))
b_x = np.vstack((x_max, -x_min))

# solve DARE
[P, K] = dare(A, B, Q, R)
A_cl = A + np.dot(B,K)

# Maximum Output Admissible Set (MOAS)
A_x_cl = np.vstack((A_x,np.dot(A_u,K)))
b_x_cl = np.vstack((b_x,b_u))
[moas_lhs, moas_rhs, t] = moas(A_cl,A_x_cl,b_x_cl)
[moas_lhs, moas_rhs] = minPolyFacets(moas_lhs, moas_rhs)
[act_plot, red_plot] = plotConsMoas(A_cl, A_x_cl, b_x_cl, t)
[moas_plot, traj_plot] = plotMoas(moas_lhs, moas_rhs, t, A_cl, 50)
plt.legend(
	[act_plot, red_plot, moas_plot, traj_plot],
    ['Non-redundant constraints',
    'First redundant constraint',
    'Maximal output admissible set',
    'Closed-loop-system trajectories'],
    loc=1)
plt.show()

# OCP blocks
[cons_lhs, cons_rhs, cons_rhs_x0] = ocpCons(A_u, b_u, A_x, b_x, moas_lhs, moas_rhs, N_ocp)
[H, F] = ocpCostFun(A, B, Q, R, P, N_ocp)

# MPC loop
x_k = x0
u_mpc = np.array([]).reshape(0,1)
x_traj_qp = np.array([]).reshape(n_x*(N_ocp+1),0)
x_traj_lqr = np.array([]).reshape(n_x,0)
for k in range(0, N_mpc):
    state_check = np.dot(moas_lhs,x_k) - moas_rhs
    if (state_check < 0).all():
        u0 = np.dot(K,x_k)
        x_traj_lqr = np.hstack((x_traj_lqr, x_k))
    else:
        u_seq = solveOcp(H, F, cons_lhs, cons_rhs, cons_rhs_x0, x_k)
        x_traj = simLinSys(x_k, N_ocp, A, B, u_seq)
        x_traj_qp = np.hstack((x_traj_qp, x_traj))
        u0 = np.reshape(u_seq[0:n_u],(n_u,1))
    u_mpc = np.vstack((u_mpc, u0))
    x_k = np.dot(A,x_k) + np.dot(B,u0)

# plot predicted trajectories
plotNomTraj(x_traj_qp, x_traj_lqr, moas_lhs, moas_rhs)
plt.show()

# plot solution
x_traj = simLinSys(x0, N_mpc, A, B, u_mpc)
plotInputSeq(u_mpc, u_min, u_max, t_s, N_mpc)
plt.show()
plotStateTraj(x_traj, x_min, x_max, t_s,N_mpc)
plt.show()
