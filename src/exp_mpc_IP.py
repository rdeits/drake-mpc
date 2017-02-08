from mpc_functions import *
from params_IP import *

# solve DARE
[P, K] = dare(A, B, Q, R)
A_cl = A + B.dot(K)

# Maximum Output Admissible Set (MOAS)
lhs_x_cl = np.vstack((lhs_x,lhs_u.dot(K)))
rhs_x_cl = np.vstack((rhs_x,rhs_u))
[lhs_moas, rhs_moas, t] = moas(A_cl, lhs_x_cl, rhs_x_cl)
[lhs_moas, rhs_moas] = minPolyFacets(lhs_moas, rhs_moas)

# OCP blocks
[G, W, E] = ocpCons(A, B, lhs_u, rhs_u, lhs_x, rhs_x, lhs_moas, rhs_moas, N_ocp)
[H, F] = ocpCostFun(A, B, Q, R, P, N_ocp)

# change variable for exeplicit MPC (z := u_seq + H^-1 F^T x0)
H_inv = np.linalg.inv(H)
S = E + G.dot(H_inv.dot(F.T))

# remove always-redundant constraints
[lhs_cons, W] = minPolyFacets(np.hstack((G, -S)), W)
G = lhs_cons[:,:n_u*N_ocp]
S = lhs_cons[:,n_u*N_ocp:]



# ################
# dt = 0.05
# A = np.array([[1, dt],
#               [0, 1]])
# b = np.array([dt**2, dt])
# H = np.array([[1.079, 0.076],
#               [0.076, 1.073]])
# F = np.array([[1.109, 1.036],
#               [1.573, 1.517]])
# G = np.array([[1, 0],
#               [0, 1],
#               [-1, 0],
#               [0, -1],
#               [dt, 0],
#               [dt, dt],
#               [-dt, 0],
#               [-dt, -dt]])
# W = np.array([[1.0], [1], [1], [1], [0.5], [0.5], [0.5], [0.5]])
# S = np.array([[1.0, 1.4],
#               [0.9, 1.3],
#               [-1.0, -1.4],
#               [-0.9, -1.3],
#               [0.1, -0.9],
#               [0.1, -0.9],
#               [-0.1, 0.9],
#               [-0.1, 0.9]])


# ############

# start from the origin
cr0 = CriticalRegion([], H, G, W, S)
L_cand = [cr0] 
L_opt = []
act_set_expl =[]

print (cr0.lhs,cr0.rhs)
if cr0.lhs.shape[0] > 2:
    plotPoly(cr0.lhs,cr0.rhs,'g')
    plt.show()

while L_cand:
    print "aaa"
    cr = L_cand[0]
    L_cand = L_cand[1:]
    act_set_expl.append(cr.act_set)
    if cr.rhs.shape[0] > 0:
        L_opt.append(cr)
        for act_set in cr.neig_act_set:
            if act_set not in act_set_expl:
                L_cand.append(CriticalRegion(act_set, H, G, W, S))
print L_opt
print L_cand
print act_set_expl