from mpc_functions import *
from params_IP import *

# qp blocks
[H, F, G, W, E] = qp_builder(A, B, Q, R, u_min, u_max, x_min, x_max, N_ocp)

# change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
H_inv = np.linalg.inv(H)
S = E + G.dot(H_inv.dot(F.T))

# start from the origin
act_set = []
cr0 = CriticalRegion(act_set, H, G, W, S)
print 'Computing critical region for the active set ' + str(act_set)
L_cand = [cr0] 
L_opt = []
act_set_cand =[cr0.act_set]

# explore the state space
while L_cand:
    # choose the first candidate in the list and remove it
    cr = L_cand[0]
    L_cand = L_cand[1:]
    # be sure that the state-space polyhedron is not empty
    if not cr.poly_t12.empty:
        # add the CR to the list of critical regions
        L_opt.append(cr)
        # compute all the potential neighboring CRs (avoid copies)
        for i in range(0,len(cr.neig_act_sets)):
            act_set = cr.neig_act_sets[i][0]
            if act_set not in act_set_cand:
                act_set_cand.append(act_set)
                licq = licq_check(G, act_set)
                if licq:
                    print 'Computing critical region for the active set ' + str(act_set)
                    L_cand.append(CriticalRegion(act_set, H, G, W, S))
                else:
                    act_set = act_set_if_degeneracy(cr, i, H, G, W, S)
                    if act_set:
                        print 'Corrected active set ' + str(act_set)
                        print 'Computing critical region for the active set ' + str(act_set)
                        L_cand.append(CriticalRegion(act_set, H, G, W, S))
                    else:
                        print "Unfeasible region detected!"

### test the explicit solution

# test point
x_test = np.array([[-1.09],[1.]])

# find the CR to which it belongs
for cr in L_opt:
	check = cr.poly_t12.lhs.dot(x_test) - cr.poly_t12.rhs
	if np.max(check) <= 0:
		cr_test = cr
		break

# derive explicit solution
z_exp = cr_test.z_opt(x_test)
u_exp = z_exp - H_inv.dot(F.T.dot(x_test))
print "Optimal solution from explicit MPC: " + str(list(u_exp.flatten()))

# solve the QP
u_impl = lin_or_quad_prog(H, (x_test.T.dot(F)).T, G, W+E.dot(x_test))[0]
print "Optimal solution from implicit MPC: " + str(list(u_impl.flatten()))

# plot the partition
for cr in L_opt:
    cr.poly_t12.plot()
plt.scatter(x_test[0], x_test[1], color='g')
plt.show()