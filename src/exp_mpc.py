from mpc_functions import *
from params_DI import *

# qp blocks
[H, F, G, W, E] = qp_builder(A, B, Q, R, u_min, u_max, x_min, x_max, N_ocp, ter_cons)

# change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
H_inv = np.linalg.inv(H)
S = E + G.dot(H_inv.dot(F.T))

# start from the origin
act_set = []
cr0 = CriticalRegion(act_set, H, G, W, S)
cr_to_be_tested = [cr0]
cr_list = []
act_sets_tested =[cr0.act_set]

# explore the state space
while cr_to_be_tested:
    # choose the first candidate in the list and remove it
    cr = cr_to_be_tested[0]
    cr_to_be_tested = cr_to_be_tested[1:]
    if not cr.poly_t12.empty:
        # add the CR to the list of critical regions
        cr_list.append(cr)
        # compute all the potential neighboring CRs (avoid copies)
        for ind in range(0, cr.poly_t12.n_fac_min):
            for act_set in cr.cand_act_sets[ind]:
                if act_set not in act_sets_tested:
                    act_sets_tested.append(act_set)
                    licq = licq_check(G, act_set)
                    if licq:
                        print 'Computing critical region for the active set ' + str(act_set)
                        cr_to_be_tested.append(CriticalRegion(act_set, H, G, W, S))
                    else:
                        act_set = degeneracy_fixer(act_set, ind, cr, H, G, W, S)
                        if act_set:
                            print 'Corrected active set ' + str(act_set)
                            cr_to_be_tested.append(CriticalRegion(act_set, H, G, W, S))
                        else:
                            print "Unfeasible region detected!"

# test the explicit solution

# find the CR to which the test point belongs
for cr in cr_list:
    check = cr.poly_t12.lhs_min.dot(x_test) - cr.poly_t12.rhs_min
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
plt.scatter(x_test[0], x_test[1], color='g')
for cr in cr_list:
    cr.poly_t12.plot2d()
plt.show()
