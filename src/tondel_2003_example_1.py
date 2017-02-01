from __future__ import absolute_import, division, print_function

import numpy as np
import pydrake.solvers.mathematicalprogram as mp


# Taking a model directly from Tondel et al. 2003, Example 1
prog = mp.MathematicalProgram()

z = prog.NewContinuousVariables(2, "z")
x = prog.NewContinuousVariables(2, "x")
prog.AddLinearConstraint(x[0] == 0)
prog.AddLinearConstraint(x[1] == 0)

dt = 0.05
A = np.array([[1, dt],
              [0, 1]])
b = np.array([dt**2, dt])
H = np.array([[1.079, 0.076],
              [0.076, 1.073]])
F = np.array([[1.109, 1.036],
              [1.573, 1.517]])
G = np.array([[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [dt, 0],
              [dt, dt],
              [-dt, 0],
              [-dt, -dt]])
W = np.array([[1.0], [1], [1], [1], [0.5], [0.5], [0.5], [0.5]])
S = np.array([[1.0, 1.4],
              [0.9, 1.3],
              [-1.0, -1.4],
              [-0.9, -1.3],
              [0.1, -0.9],
              [0.1, -0.9],
              [-0.1, 0.9],
              [-0.1, 0.9]])

for i in range(G.shape[0]):
    prog.AddLinearConstraint(G[i,:].dot(z) <= W[i,0] + S[i,:].dot(x))

prog.AddQuadraticCost(H, np.zeros(z.size), z)


def get_objective_matrix(prog, vars, params):
    H = np.zeros((prog.num_vars(), prog.num_vars()))
    for binding in prog.quadratic_costs():
        print("vars", binding.variables())
        indices = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        print("Q:", binding.constraint().Q())
        print("indices:", indices)
        Q = binding.constraint().Q()
        for (iQ, iH) in enumerate(indices):
            H[iH, iH] = Q[iQ, iQ]
    for p in params:
        assert all(H[prog.FindDecisionVariableIndex(p), :] == 0)
        assert all(H[:, prog.FindDecisionVariableIndex(p)] == 0)

    v_inds = np.array([prog.FindDecisionVariableIndex(v) for v in vars])
    H = H[v_inds, v_inds.reshape((-1, 1))]
    H = H + H.T - np.diag(np.diag(H))
    return H

def get_linconstr_matrices(prog, vars, params):
    A_elements = []
    lb_elements = []
    ub_elements = []
    for binding in prog.linear_constraints():
        constraint = binding.constraint()
        ai = constraint.A()
        ai_full = np.zeros((ai.shape[0], prog.num_vars()))
        indices = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        for (iconstr, ivar) in enumerate(indices):
            ai_full[:,ivar] = ai[:,iconstr]
        A_elements.append(ai_full)
        lb_elements.append(constraint.lower_bound())
        ub_elements.append(constraint.upper_bound())
    A = np.vstack(A_elements)
    lb = np.vstack(lb_elements)
    ub = np.vstack(ub_elements)
    G = A[:, [prog.FindDecisionVariableIndex(v) for v in vars]]
    S = -A[:, [prog.FindDecisionVariableIndex(p) for p in params]]
    W = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        if lb[i] == -np.inf:
            W[i] = ub[i]
        else:
            assert ub[i] == np.inf, "Only one-sided linear inequalities are supported"
            W[i] = -lb[i]
            G[i, :] *= -1
            S[i, :] *= -1
    return G, W, S


print("H:", get_objective_matrix(prog, z, x))
G, W, S = get_linconstr_matrices(prog, z, x)
print("G:", G)
print("W:", W)
print("S:", S)


result = prog.Solve()
assert result == mp.SolutionResult.kSolutionFound

print(prog.GetSolution(z))
