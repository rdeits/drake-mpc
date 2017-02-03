from __future__ import absolute_import, division, print_function

from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import pydrake.solvers.mathematicalprogram as mp
import irispy


def objective_matrix(prog, vars, params):
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

def linconstr_matrices(prog, vars, params):
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

def linearly_independent_subset(G, active_indices):
    num_constraints = min(len(active_indices), G.shape[1])
    if num_constraints == 0:
        return []
    for Ai in combinations(active_indices, num_constraints):
        GA = G[Ai,:]
        if np.linalg.matrix_rank(GA) == num_constraints:
            return Ai
    raise ValueError("No linearly independent subset of rows could be found")


def critical_region(model, vars, params):
    H = objective_matrix(prog, vars, params)
    z = prog.GetSolution(vars)
    x = prog.GetSolution(params)
    G, W, S = linconstr_matrices(prog, vars, params)

    constraint_values = G.dot(z) - W - S.dot(x)
    active_constraint_indices = [i for (i, v) in enumerate(constraint_values) \
        if np.isclose(v, 0)]
    print("active indices:", active_constraint_indices)
    Ai = linearly_independent_subset(G, active_constraint_indices)
    GA = G[Ai, :]
    WA = W[Ai]
    SA = S[Ai, :]
    print("H:", H)
    print("GA:", GA)

    crmodel = mp.MathematicalProgram()
    crvars = crmodel.NewContinuousVariables(len(params), "crvars")

    if Ai:
        lambdaA = -np.linalg.inv(GA.dot(np.linalg.inv(H)).dot(GA.T)).dot(WA + SA.dot(crvars))
        z = -np.linalg.inv(H).dot(GA.T).dot(lambdaA)
    constr_expr = G.dot(z) - S.dot(crvars) - W
    for expr in constr_expr:
        print("constraint:", expr <= 0)
        crmodel.AddLinearConstraint(expr <= 0)
    return crmodel, crvars

def feasible_set_polyhedron(model, vars):
    G, W, S = linconstr_matrices(model, vars, [])
    assert S.size == 0
    p = irispy.Polyhedron(G, W)
    return p



if __name__ == '__main__':
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

    result = prog.Solve()
    assert result == mp.SolutionResult.kSolutionFound

    crmodel, crvars = critical_region(prog, z, x)
    p = feasible_set_polyhedron(crmodel, crvars)
    pts = np.vstack(p.generatorPoints())

    plt.plot(pts[[0,1,2,3,0],0], pts[[0,1,2,3,0],1], "r.-")
    plt.show()

