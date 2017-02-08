from mpc_functions import *
from params_IP import *

# initial state
x0 = np.array([[1.],[0.]])

# solve DARE
[P, K] = dare(A, B, Q, R)
A_cl = A + B.dot(K)

# Maximum Output Admissible Set (MOAS)
lhs_x_cl = np.vstack((lhs_x,lhs_u.dot(K)))
rhs_x_cl = np.vstack((rhs_x,rhs_u))
[lhs_moas, rhs_moas, t] = moas(A_cl, lhs_x_cl, rhs_x_cl)
[lhs_moas, rhs_moas] = minPolyFacets(lhs_moas, rhs_moas)
[act_plot, red_plot] = plotConsMoas(A_cl, lhs_x_cl, rhs_x_cl, t)
[moas_plot, traj_plot] = plotMoas(lhs_moas, rhs_moas, t, A_cl, 50)
plt.legend(
	[act_plot, red_plot, moas_plot, traj_plot],
    ['Non-redundant constraints',
    'First redundant constraint',
    'Maximal output admissible set',
    'Closed-loop-system trajectories'],
    loc=1)
plt.show()

# OCP blocks
[G, W, E] = ocpCons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_moas, rhs_moas, N_ocp)
[H, F] = ocpCostFun(A, B, Q, R, P, N_ocp)

# MPC loop
x_k = x0
u_mpc = np.array([]).reshape(0,1)
x_traj_qp = np.array([]).reshape(n_x*(N_ocp+1),0)
x_traj_lqr = np.array([]).reshape(n_x,0)
for k in range(0, N_mpc):
    state_check = lhs_moas.dot(x_k) - rhs_moas
    if (state_check < 0).all():
        u0 = K.dot(x_k)
        x_traj_lqr = np.hstack((x_traj_lqr, x_k))
    else:
        u_seq = solveOcp(H, F, G, W, E, x_k)
        x_traj = simLinSys(x_k, N_ocp, A, B, u_seq)
        x_traj_qp = np.hstack((x_traj_qp, x_traj))
        u0 = u_seq[0:n_u].reshape(n_u,1)
    u_mpc = np.vstack((u_mpc, u0))
    x_k = A.dot(x_k) + B.dot(u0)

# plot predicted trajectories
plotNomTraj(x_traj_qp, x_traj_lqr, lhs_moas, rhs_moas)
plt.show()

# plot solution
x_traj = simLinSys(x0, N_mpc, A, B, u_mpc)
plotInputSeq(u_mpc, u_min, u_max, t_s, N_mpc)
plt.show()
plotStateTraj(x_traj, x_min, x_max, t_s,N_mpc)
plt.show()