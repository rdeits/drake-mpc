from params_DI import *
from mpc_functions import *
from plot_functions import *

# solve DARE
[P, K] = dare(A, B, Q, R)
A_cl = A + B.dot(K)

# OCP constraints
lhs_u = np.vstack((np.eye(n_u), -np.eye(n_u)))
rhs_u = np.vstack((u_max, -u_min))
lhs_x = np.vstack((np.eye(n_x), -np.eye(n_x)))
rhs_x = np.vstack((x_max, -x_min))

# Maximum Output Admissible Set (MOAS)
lhs_x_cl = np.vstack((lhs_x, lhs_u.dot(K)))
rhs_x_cl = np.vstack((rhs_x, rhs_u))
[lhs_moas, rhs_moas, t] = moas(A_cl, lhs_x_cl, rhs_x_cl)
poly_moas = Poly(lhs_moas, rhs_moas)
[lhs_moas, rhs_moas] = [poly_moas.lhs_min, poly_moas.rhs_min]

# QP blocks
[G, W, E] = ocp_cons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_moas, rhs_moas, N_ocp)
[H, F] = ocp_cost_fun(A, B, Q, R, P, N_ocp)

# MPC loop
xk = x0
u_mpc = np.array([]).reshape(0,1)
x_traj_qp = np.array([]).reshape(n_x*(N_ocp+1),0)
x_traj_lqr = np.array([]).reshape(n_x,0)
for k in range(0, N_mpc):
    state_check = lhs_moas.dot(xk) - rhs_moas
    if (state_check < 0).all():
        u0 = K.dot(xk)
        x_traj_lqr = np.hstack((x_traj_lqr, xk))
    else:
        u_seq = lin_or_quad_prog(H, (xk.T.dot(F)).T, G, W+E.dot(xk))[0]
        x_traj = sim_lin_sys(xk, N_ocp, A, B, u_seq)
        x_traj_qp = np.hstack((x_traj_qp, x_traj))
        u0 = u_seq[0:n_u].reshape(n_u,1)
    u_mpc = np.vstack((u_mpc, u0))
    xk = A.dot(xk) + B.dot(u0)

# plot moas
[act_plot, red_plot] = plot_cons_moas(A_cl, lhs_x_cl, rhs_x_cl, t)
[moas_plot, traj_plot] = plot_moas(lhs_moas, rhs_moas, t, A_cl, 50)
plt.legend(
    [act_plot, red_plot, moas_plot, traj_plot],
    ['Non-redundant constraints',
    'First redundant constraint',
    'Maximal output admissible set',
    'Closed-loop-system trajectories'],
    loc=1)
plt.show()

# plot predicted trajectories
plot_nom_traj(x_traj_qp, x_traj_lqr, lhs_moas, rhs_moas)
plt.show()

# plot mpc trajectory
x_traj = sim_lin_sys(x0, N_mpc, A, B, u_mpc)
plot_input_seq(u_mpc, u_min, u_max, t_s, N_mpc)
plt.show()
plot_state_traj(x_traj, x_min, x_max, t_s,N_mpc)
plt.show()