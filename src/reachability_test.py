from params_IP import *
from mpc_functions import *

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

# reachability set
lhs_reach_set = np.array([]).reshape(0,n_x+n_u*N_ocp)
rhs_reach_set = np.array([]).reshape(0,1)

# constraint each input
lhs_all_u = la.block_diag(*[lhs_u for i in range(0, N_ocp)])
rhs_all_u = np.vstack([rhs_u for i in range(0, N_ocp)])

# constraint initial state
lhs_reach_set = np.vstack((lhs_reach_set, la.block_diag(lhs_x, lhs_all_u)))
rhs_reach_set = np.vstack((rhs_reach_set, np.vstack((rhs_x, rhs_all_u))))

# constraint each state
[for_evo, free_evo] = lin_sys_evo(A, B, N_ocp)
evo = np.hstack((free_evo, for_evo))
for i in range(0,N_ocp):
    lhs_reach_set = np.vstack((lhs_reach_set,lhs_x.dot(evo[n_x*i:n_x*(i+1),:])))
    rhs_reach_set = np.vstack((rhs_reach_set, rhs_x))

# constraint final state
    if i == N_ocp-1:
        lhs_reach_set = np.vstack((lhs_reach_set,lhs_moas.dot(evo[n_x*i:n_x*(i+1),:])))
        rhs_reach_set = np.vstack((rhs_reach_set, rhs_moas))

# plot
state_poly = Poly(lhs_x, rhs_x)
reach_poly = Poly(lhs_reach_set[2*n_x:,:], rhs_reach_set[2*n_x:,:])
state_poly.plot2d('g')
poly_moas.plot2d('r')
reach_poly.plot2d('b', [0,1])
plt.show()