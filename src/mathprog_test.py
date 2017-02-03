from pydrake.solvers import mathematicalprogram as mp
import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt

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

# def msas(A,A_x,b_x):
#     """
#     msas(A,A_x,b_x):
#     returns the maximum state admissibile set (i.e., the set such that
#     if x(0) \in msas then A_x*x(k) <= b_x with x(k)=A*x(k-1) \forall k>=1)
#     INPUTS:
#     A -> state transition matrix
#     [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
#     OUTPUTS:
#     [msasLhs, msasRhs] -> matrices such that msas := {x | msasLhs*x <= msasRhs}
#     t                  -> number of steps after that the constraints are redundant
#     """
#     n_x = np.shape(A)[0]
#     t = 0
#     convergence = False
#     while convergence == False:
#         # set bounds to all variables to ensure the finiteness of the solution
#         ub = 1e6*np.ones(n_x,1)
#         lb = -ub
#         # number of state constraints
#         s = np.shape(consRhs)[0]
#         # cost function jacobians for all i
#         J = - np.dot(A_x, np.linalg.matrix_power(A,t+1))
#         # constraints to each LP
#         consLhs = np.vstack([np.dot(A_x, np.linalg.matrix_power(A,k)) for k in range(0,t+1)])
#         consRhs = np.vstack([b_x for k in range(0,t+1)])
#         # assemble LP
#         prog = mp.MathematicalProgram()
#         x = prog.NewContinuousVariables(n_x, "x")
#         # impose constraints
#         for j in range(0, np.shape(consRhs)[0]):
#             prog.AddLinearConstraint(np.dot(consLhs[j,:], x) <= consRhs[j])
#         # impose bounds
#         prog.AddLinearConstraint(x <= ub)
#         prog.AddLinearConstraint(x >= lb)
#         for i in range(1,s+1):
#             # cost function
#             cost = prog.AddLinearCost(np.dot(J[i-1],x))
#             # solve LP
#             prog.Solve()
#             J_sol_i = ??? + b_x[i]
#             if J_sol_i > 0:
#                 t += 1
#                 break
#         convergence = True
#     msasLhs = consLhs
#     msasRhs = consRhs
#     return [msasLhs, msasRhs, t]





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

def ocpCons(A_u,b_u,A_x,b_x,N):
    """
    ocpCons(A_u,b_u,A_x,b_x,N):
    returns the constraint blocks of the ocp QP
    INPUTS:
    [A_u, b_u] -> input constraint A_u u(k) <= b_u \forall k
    [A_x, b_x] -> state constraint A_x x(k) <= b_x \forall k
    N          -> time steps
    OUTPUTS:
    [inLhs, inRhs]           -> QP constraints such that inLhs*u_seq <= inRhs
    [stLhs, stRhs, stRhs_x0] -> QP constraints such that stLhs*u_seq <= stRhs + stRhs_x0*x0
    """
    # input constraints
    inLhs = la.block_diag(*[A_u for k in range(0, N)])
    inRhs = np.vstack([b_u for k in range(0, N)])
    # evolution of the system
    [forEvo, freeEvo] = linSysEvo(A,B,N)
    # output constraints
    diagA_x = la.block_diag(*[A_x for k in range(1, N+1)])
    stLhs = np.dot(diagA_x, forEvo)
    stRhs = np.vstack([b_x for k in range(1, N+1)])
    stRhs_x0 = - np.dot(diagA_x, freeEvo)
    return [inLhs, inRhs, stLhs, stRhs, stRhs_x0]

def simLinSys(A,B,u_seq,x0):
    """
    simLinSys(A,B,u_seq,x0):
    simulates the evolution of the linear system
    INPUTS:
    A     -> state transition matrix
    B     -> input to state matrix
    u_seq -> input sequence \in R^(N*n_u)
    x0    -> initial conditions
    OUTPUTS:
    [inLhs, inRhs]           -> QP constraints such that inLhs*u_seq <= inRhs
    [stLhs, stRhs, stRhs_x0] -> QP constraints such that stLhs*u_seq <= stRhs + stRhs_x0*x0
    """
    # system dimensions
    n_x = np.shape(A)[1]
    n_u = np.shape(B)[1]
    N = np.shape(u_seq)[0]/n_u
    # state trajectory
    x_traj = x0
    for k in range(1, N+1):
        x_prev = x_traj[n_x*(k-1):n_x*k]
        u = u_seq[n_u*(k-1):n_u*k]
        x_traj = np.vstack((x_traj, np.dot(A,x_prev) + np.dot(B,u)))
    return x_traj

def plotInputSeq(u_seq, u_min, u_max, t_s, N):
    """
    plotInputSeq(x,t_s):
    plots the input sequences as functions of time
    INPUTS:
    u_seq -> input sequence \in R^(N*n_u)
    t_s   -> sampling time
    N     -> time steps
    title -> title of the plot
    """
    n_u = np.shape(u_seq)[0]/N
    u_seq = np.reshape(u_seq,(n_u,N), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        plt.step(t, np.hstack((u_seq[i,0],u_seq[i,:])),'b')
        plt.step(t, u_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, u_max[i,0]*np.ones(np.shape(t)),'r')
        plt.ylabel(r'u'+str(i+1))
        plt.xlim((0.,N*t_s))
    plt.xlabel('t')
    plt.show()

def plotStateTraj(x_traj, x_min, x_max, t_s, N):
    """
    plotStateTraj(x,t_s):
    plots the state trajectories as functions of time
    INPUTS:
    x_traj -> state trajectory \in R^((N+1)*n_x)
    t_s    -> sampling time
    N      -> time steps
    title  -> title of the plot
    """
    n_x = np.shape(x_traj)[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        plt.plot(t, x_traj[i,:],'b')
        plt.step(t, x_min[i,0]*np.ones(np.shape(t)),'r')
        plt.step(t, x_max[i,0]*np.ones(np.shape(t)),'r')
        plt.ylabel(r'x'+str(i+1))
        plt.xlim((0.,N*t_s))
    plt.xlabel('t')
    plt.show()

# dynamic parameters
m = 1.
l = 1.
g = 10

# initial state
x0 = np.array([[1],[0.]])

# system dymensions
n_x = 2
n_u = 1

# discretization
t_s = .1
N_ocp = 5

# ocp cost function
Q = np.eye(n_x)
R = np.eye(n_u)

# ocp bounds
u_max = np.array([[12]])
u_min = -u_max
x_max = np.array([[1],[1]])
x_min = -x_max

# continuous time dynamics
A = np.array([[0, 1], [g/l, 0]])
B = np.array([[0], [1/(m**l)]])
C = np.eye(n_x)
n_y = np.shape(C)[0]
D = np.zeros((n_y,n_u))

# discretization
sys = sig.cont2discrete((A,B,C,D),t_s,'zoh')
[A, B, C, D] = sys[0:4]

# solve dare
[P, K] = dare(A,B,Q,R)

# ocp blocks
[H, F] = ocpCostFun(A,B,Q,R,P,N_ocp)
A_u = np.vstack((np.eye(n_u), -np.eye(n_u)))
A_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
b_u = np.vstack((u_max, -u_min))
b_x = np.vstack((x_max, -x_min))
[inLhs, inRhs, stLhs, stRhs, stRhs_x0] = ocpCons(A_u,b_u,A_x,b_x,N_ocp)

# mpc
N_mpc = 50
u_mpc = np.zeros((n_u*N_mpc,1))
x_k = x0
for k in range(0, N_mpc):
    prog = mp.MathematicalProgram()
    u_seq = prog.NewContinuousVariables(n_u*N_ocp, "u_seq")
    prog.AddQuadraticCost(H, np.dot(F,x_k), u_seq)
    for i in range(0, np.shape(inRhs)[0]):
        prog.AddLinearConstraint(np.dot(inLhs[i,:], u_seq) <= inRhs[i])
    for i in range(0, np.shape(stRhs)[0]):
        prog.AddLinearConstraint(np.dot(stLhs[i,:], u_seq) <= stRhs[i] + np.dot(stRhs_x0[i,:], x_k))
    result = prog.Solve()
    print(result)
    u0 = np.reshape(prog.GetSolution(u_seq)[0:n_u],(n_u,1))
    u_mpc[k*n_u:(k+1)*n_u,0] = u0
    x_k = np.dot(A,x_k) + np.dot(B,u0)

# plot solution
x_traj = simLinSys(A,B,u_mpc,x0)
plotInputSeq(u_mpc,u_min,u_max,t_s,N_mpc)
plotStateTraj(x_traj,x_min,x_max,t_s,N_mpc)


