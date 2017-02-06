from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.spatial as spat
import matplotlib as mpl
import matplotlib.pyplot as plt
import irispy

def dare(A,B,Q,R):
    """
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
    P = la.solve_discrete_are(A,B,Q,R)
    K = -np.dot(np.dot(np.dot(la.inv(np.dot(np.dot(np.transpose(B),P),B)+R),np.transpose(B)),P),A)
    return [P, K]

def minPolyFacets(A,b):
    """
    minPolyFacets(A,b):
    removes all the redundant constraints from a polytope
    INPUTS:
    [A, b] -> polytope definition {x | A*x <= b}
    OUTPUTS:
    [A_min, b_min] -> minimal polytope definition
    """
    # buond the problem
    n = np.shape(A)[1]
    ub = 1e6*np.ones((n,1))
    lb = -ub
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
            prog.AddLinearConstraint(x[j] <= ub[j])
            prog.AddLinearConstraint(x[j] >= lb[j])
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

def moas(A,A_x,b_x):
    """
    moas(A,A_x,b_x):
    returns the maximum output admissibile set (i.e., the set such that
    if x(0) \in moas then A_x*x(k) <= b_x with x(k)=A*x(k-1) \forall k>=1)
    Algorithm 3.2 from "Gilbert, Tan - Linear Systems with State and Control Constraints:
    The Theory and Application of Maximal Output Admissible Sets"
    INPUTS:
    A -> state transition matrix
    [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
    OUTPUTS:
    [moasLhs, moasRhs] -> matrices such that moas := {x | moasLhs*x <= moasRhs}
    t                  -> number of steps after that the constraints are redundant
    """
    n_x = np.shape(A)[0]
    t = 0
    convergence = False
    while convergence == False:
        # set bounds to all variables to ensure the finiteness of the solution
        ub = 1e6*np.ones((n_x,1))
        lb = -ub
        # cost function jacobians for all i
        J = np.dot(A_x, np.linalg.matrix_power(A,t+1))
        # constraints to each LP
        consLhs = np.vstack([np.dot(A_x, np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
        consRhs = np.vstack([b_x for k in range(0,t+1)])
        # number of state constraints
        s = np.shape(b_x)[0]
        # assemble LP
        prog = mp.MathematicalProgram()
        x = prog.NewContinuousVariables(n_x, "x")
        # impose constraints
        for j in range(0, np.shape(consRhs)[0]):
            prog.AddLinearConstraint(np.dot(consLhs[j,:], x) <= consRhs[j])
        # impose bounds
        for j in range(0,n_x):
            prog.AddLinearConstraint(x[j] <= ub[j])
            prog.AddLinearConstraint(x[j] >= lb[j])
        # vector of all minima
        J_sol = np.zeros(s)
        for i in range(1,s+1):
            # cost function
            prog.AddLinearCost(np.dot(-J[i-1],x))
            # solve LP
            solver = GurobiSolver()
            result = solver.Solve(prog)
            J_sol[i-1] = np.dot(J[i-1],prog.GetSolution(x)) - b_x[i-1]
        if np.max(J_sol) < 0:
            convergence = True
        else:
            t += 1
    moasLhs = consLhs
    moasRhs = consRhs
    return [moasLhs, moasRhs, t]

def linSysEvo(A,B,N):
    """
    linSysEvo(A,B,N):
    returns the free and froced evolution matrices
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    N -> time steps
    OUTPUTS:
    [forEvo, freeEvo] -> matrices such that x_traj = forEvo*u_seq + freeEvo*x_0
    """
    # forced evolution of the system
    [n_x, n_u] = np.shape(B)
    forEvo = np.zeros((n_x*N,n_u*N))
    for i in range(0, N):
        for j in range(0, i+1):
            forEvo_ij = np.dot(np.linalg.matrix_power(A,i-j),B)
            forEvo[n_x*i:n_x*(i+1),n_u*j:n_u*(j+1)] = forEvo_ij
    # free evolution of the system
    freeEvo = np.vstack([np.linalg.matrix_power(A,k) for k in range(1, N+1)])
    return [forEvo, freeEvo]

def ocpCostFun(A,B,Q,R,P,N):
    """
    ocpCostFun(A,B,Q,R,P,N):
    returns the cost function blocks of the ocp QP
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    Q -> state weight matrix of the LQR problem
    R -> input weight matrix of the LQR problem
    N -> time steps
    OUTPUTS:
    qCost -> Hessian of the ocp QP
    lCost -> cost function linear term (to be right-multiplied by x_0!)
    """
    # quadratic term in the state sequence
    qSt = la.block_diag(*[Q for k in range(0, N-1)])
    qSt = la.block_diag(qSt,P)
    # quadratic term in the input sequence
    qIn = la.block_diag(*[R for k in range(0, N)])
    # evolution of the system
    [forEvo, freeEvo] = linSysEvo(A,B,N)
    # quadratic term
    qCost = 2*(qIn+np.dot(np.transpose(forEvo),np.dot(qSt,forEvo)))
    # linear term
    lCost = 2*np.dot(np.dot(np.transpose(forEvo),np.transpose(qSt)),freeEvo)
    return [qCost,lCost]

def ocpCons(A_u,b_u,A_x,b_x,A_xN,b_xN,N):
    """
    ocpCons(A_u,b_u,A_x,b_x,N):
    returns the constraint blocks of the ocp QP
    INPUTS:
    [A_u, b_u]   -> input constraint A_u u(k) <= b_u \forall k
    [A_x, b_x]   -> state constraint A_x x(k) <= b_x \forall k
    [A_xf, b_xf] -> final state constraint A_xN x(N) <= b_xN
    N            -> time steps
    OUTPUTS:
    [inLhs, inRhs]              -> QP constraints such that inLhs*u_seq <= inRhs
    [stLhs, stRhs, stRhs_x0]    -> QP constraints such that stLhs*u_seq <= stRhs + stRhs_x0*x0
    [terLhs, terRhs, terRhs_x0] -> QP constraints such that terLhs*u_seq <= terRhs + terRhs_x0*x0
    """
    # input constraints
    inLhs = la.block_diag(*[A_u for k in range(0, N)])
    inRhs = np.vstack([b_u for k in range(0, N)])
    # evolution of the system
    [forEvo, freeEvo] = linSysEvo(A,B,N)
    # state constraints
    diagA_x = la.block_diag(*[A_x for k in range(1, N+1)])
    stLhs = np.dot(diagA_x, forEvo)
    stRhs = np.vstack([b_x for k in range(1, N+1)])
    stRhs_x0 = - np.dot(diagA_x, freeEvo)
    # terminal constraints
    n_x = np.shape(A_x)[1]
    forEvoTer = forEvo[-n_x:,:]
    terLhs = np.dot(A_xN,forEvoTer)
    terRhs = b_xN
    terRhs_x0 = - np.dot(A_xN,np.linalg.matrix_power(A,N))
    return [inLhs, inRhs, stLhs, stRhs, stRhs_x0, terLhs, terRhs, terRhs_x0]

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
        lbPlot, = plt.step(t, u_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, u_max[i,0]*np.ones(np.shape(t)),'r')
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend([inPlot, lbPlot],['Optimal control', 'Control bounds'], loc=1)
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
        lbPlot, = plt.step(t, x_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, x_max[i,0]*np.ones(np.shape(t)),'r')
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend([stPlot, lbPlot],['Optimal trajectory', 'State bounds'], loc=1)
    plt.xlabel(r'$t$')

def plotStateSpaceTraj(x_traj, N, col='b'):
    """
    plotStateTraj(x_traj,n_x):
    plots the state trajectories as functions of time (ONLY 2D)
    INPUTS:
    x_traj -> state trajectory \in R^((N+1)*n_x)
    N      -> time steps
    OUTPUTS:
    trajPlot -> figure handle
    """
    n_x = np.shape(x_traj)[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    plt.scatter(x_traj[0,0], x_traj[1,0], color=col, alpha=.5)
    trajPlot = plt.plot(x_traj[0,:], x_traj[1,:], color=col)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    return trajPlot

def plotPoly(A,b,col):
    """
    plotPoly(A,b,col):
    plots a polytope (ONLY 2D)
    INPUTS:
    [A, b] -> polytope A*x <= b
    col    -> line specs
    OUTPUTS:
    polyPlot -> figure handle
    """
    poly = irispy.Polyhedron(A, b)
    verts = np.vstack(poly.generatorPoints())
    verts = np.vstack((verts,verts[-1,:]))
    hull = spat.ConvexHull(verts)
    for simp in hull.simplices:
        polyPlot, = plt.plot(verts[simp, 0], verts[simp, 1], col)
    return polyPlot

def plotActConsMoas(A,A_x,b_x,t):
    """
    plotActConsMoas(A,A_x,b_x,t,leg):
    plots the set of constraints that are not redundant in the definition of the moas
    INPUTS:
    A          -> state transition matrix
    [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
    t          -> number of steps after that the constraints are redundant
    OUTPUTS:
    [actPlot, redPlot] -> figure handles
    """
    # plot constraints until the first redundant
    for i in range(t+1,-1,-1):
        A_x_i = np.dot(A_x, np.linalg.matrix_power(A,i))
        if i == t+1:
            redPlot = plotPoly(A_x_i, b_x, 'g-.')
        else:
            actPlot = plotPoly(A_x_i, b_x, 'y-.')
    return [actPlot, redPlot]

def plotMoas(moasLhs, moasRhs, t, A, N):
    """
    plotMoas(moasLhs, moasRhs, t, A_cl, B_cl, N):
    plots the maximum output admissible set and a trajectory for each vertex of the moas
    INPUTS:
    [moasLhs, moasRhs] -> matrices such that moas := {x | moasLhs*x <= moasRhs}
    A                  -> state transition matrix
    t                  -> number of steps after that the constraints are redundant
    N                  -> number of steps for the simulations
    OUTPUTS:
    [moasPlot, trajPlot] -> figure handles
    """
    # plot moas polyhedron
    moasPlot = plotPoly(moasLhs, moasRhs, 'r-')
    # simulate a trajectory for each vertex
    poly = irispy.Polyhedron(moasLhs, moasRhs)
    verts = np.vstack(poly.generatorPoints())
    for i in range(0, np.shape(verts)[0]):
        vert = np.reshape(verts[i,:],(n_x,1))
        x_traj = simLinSys(vert, N, A)
        trajPlot, = plotStateSpaceTraj(x_traj,N)
    return [moasPlot, trajPlot]

def plotNomTraj(x_traj_qp, moasLhs, moasRhs):
    """
    plotNomTraj(x_traj_qp, moasLhs, moasRhs):
    plots the open-loop optimal trajectories for each sampling time (ONLY 2D)
    INPUTS:
    x_traj_qp -> matrix with optimal trajectories
    [moasLhs, moasRhs] -> matrices such that moas := {x | moasLhs*x <= moasRhs}
    """
    n_traj = np.shape(x_traj_qp)[1]
    N_ocp = (np.shape(x_traj_qp)[0]-1)/2
    colMap = plt.get_cmap('jet')
    colNorm  = mpl.colors.Normalize(vmin=0, vmax=n_traj)
    scalarMap = mpl.cm.ScalarMappable(norm=colNorm, cmap=colMap)
    polyPlot = plotPoly(moasLhs, moasRhs, 'r-')
    legPlot = [polyPlot]
    legLab = ['MOAS']
    for i in range(0,n_traj-1):
        col = scalarMap.to_rgba(i)
        trajPlot = plotStateSpaceTraj(x_traj_qp[:,i], N_ocp, col)
        legPlot += trajPlot
        legLab += [r'$\mathbf{x}^*(x(t=' + str(i*t_s) + ' \mathrm{s}))$']
    plt.legend(legPlot, legLab, loc=1)

def mpcController(A, B, Q, R, A_u, b_u, A_x, b_x, x0, N_mpc):
    """
    mpcController(A, B, Q, R, A_u, b_u, A_x, b_x, x0, N_mpc):
    simulates the closed-loop system with the MPC controller
    INPUTS:
    A -> state transition matrix
    B -> input to state matrix
    Q -> state weight matrix of the LQR problem
    R -> input weight matrix of the LQR problem
    [A_u, b_u]   -> input constraint A_u u(k) <= b_u \forall k
    [A_x, b_x]   -> state constraint A_x x(k) <= b_x \forall k
    x0 -> initial conditions
    N_mpc -> MPC controller horizon
    """
    # solve dare
    [P, K] = dare(A,B,Q,R)
    A_cl = A + np.dot(B,K)
    # maximum output admissible set
    A_x_cl = np.vstack((A_x,np.dot(A_u,K)))
    b_x_cl = np.vstack((b_x,b_u))
    [moasLhs, moasRhs, t] = moas(A_cl,A_x_cl,b_x_cl)
    [moasLhs, moasRhs] = minPolyFacets(moasLhs, moasRhs)
    # plot moas
    [actPlot, redPlot] = plotActConsMoas(A_cl, A_x_cl, b_x_cl, t)
    [moasPlot, trajPlot] = plotMoas(moasLhs, moasRhs, t, A_cl, 100)
    plt.legend([actPlot, redPlot, moasPlot, trajPlot],
        ['Non-redundant constraints','First redundant constraint','Maximal output admissible set','Closed-loop-system trajectories'])
    plt.show()
    # ocp blocks
    [inLhs, inRhs, stLhs, stRhs, stRhs_x0, terLhs, terRhs, terRhs_x0] = ocpCons(A_u, b_u, A_x, b_x, moasLhs, moasRhs, N_ocp)
    [H, F] = ocpCostFun(A,B,Q,R,P,N_ocp)
    # mpc loop
    u_mpc = np.zeros((n_u*N_mpc,1))
    x_k = x0
    x_traj_qp = np.array([]).reshape(n_x*(N_ocp+1),0)
    for k in range(0, N_mpc):
        # check if the state is in moas
        stateInMoas = np.dot(moasLhs,x_k) - moasRhs
        if (stateInMoas < 0).all():
            u0 = np.dot(K,x_k)
        else:
            prog = mp.MathematicalProgram()
            u_seq = prog.NewContinuousVariables(n_u*N_ocp, "u_seq")
            prog.AddQuadraticCost(H, np.dot(F,x_k), u_seq)
            consLhs = np.vstack((inLhs, stLhs, terLhs))
            consRhs = np.vstack((inRhs, stRhs + np.dot(stRhs_x0,x_k),  terRhs + np.dot(terRhs_x0,x_k)))
            for i in range(0, np.shape(consRhs)[0]):
                prog.AddLinearConstraint(np.dot(consLhs[i,:], u_seq) <= consRhs[i])
            solver = GurobiSolver()
            result = solver.Solve(prog)
            u0 = np.reshape(prog.GetSolution(u_seq)[0:n_u],(n_u,1))
            u_seq_opt = np.reshape(prog.GetSolution(u_seq),(n_u*N_ocp,1))
            x_traj = simLinSys(x_k, N_ocp, A, B, u_seq_opt)
            x_traj_qp = np.hstack((x_traj_qp,x_traj))
        u_mpc[k*n_u:(k+1)*n_u,0] = u0
        x_k = np.dot(A,x_k) + np.dot(B,u0)
    # plot predicted trajectories
    plotNomTraj(x_traj_qp, moasLhs, moasRhs)
    plt.show()
    # plot solution
    x_traj = simLinSys(x0, N_mpc, A, B, u_mpc)
    plotInputSeq(u_mpc,u_min,u_max,t_s,N_mpc)
    plt.show()
    plotStateTraj(x_traj,x_min,x_max,t_s,N_mpc)
    plt.show()


# dynamic parameters
m = 1.
l = 1.
g = 10

# initial state
x0 = np.array([[1],[0.]])

# discretization
t_s = .1
N_ocp = 5
N_mpc = 50

# system dynamics
A = np.array([[0, 1], [g/l, 0]])
B = np.array([[0], [1/(m**l)]])
[n_x, n_u] = np.shape(B)
C = np.eye(n_x)
n_y = np.shape(C)[0]
D = np.zeros((n_y,n_u))
sys = sig.cont2discrete((A,B,C,D),t_s,'zoh')
[A, B, C, D] = sys[0:4]

# ocp cost function
Q = np.eye(n_x)
R = np.eye(n_u)

# ocp bounds
u_max = np.array([[12]])
u_min = -u_max
x_max = np.array([[1],[1]])
x_min = -x_max

# ocp constraints
A_u = np.vstack((np.eye(n_u), -np.eye(n_u)))
A_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
b_u = np.vstack((u_max, -u_min))
b_x = np.vstack((x_max, -x_min))

# test controller
mpcController(A, B, Q, R, A_u, b_u, A_x, b_x, x0, N_mpc)