import itertools
import numpy as np
import pydrake.solvers.mathematicalprogram as mp


def extract_linear_equalities(prog):
    bindings = prog.linear_equality_constraints()
    C = np.zeros((len(bindings), prog.num_vars()))
    d = np.zeros(len(bindings))
    for (i, binding) in enumerate(bindings):
        constraint = binding.constraint()
        ci = np.zeros(prog.num_vars())
        assert constraint.upper_bound() == constraint.lower_bound()
        d[i] = constraint.upper_bound()
        ai = constraint.A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for (j, var) in enumerate(binding.variables()):
            ci[prog.FindDecisionVariableIndex(var)] = ai[j]
        C[i, :] = ci
    return C, d


def extract_linear_inequalities(prog):
    bindings = itertools.chain(prog.linear_constraints(), prog.bounding_box_constraints())
    if not bindings:
        return np.zeros((0, prog.num_vars())), np.zeros(0)
    A = []
    b = []
    for (i, binding) in enumerate(bindings):
        constraint = binding.constraint()
        A_row = np.zeros(prog.num_vars())
        ai = constraint.A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for (j, var) in enumerate(binding.variables()):
            A_row[prog.FindDecisionVariableIndex(var)] = ai[j]

        if constraint.upper_bound() != np.inf:
            A.append(A_row)
            b.append(constraint.upper_bound())
        if constraint.lower_bound() != -np.inf:
            A.append(-A_row)
            b.append(-constraint.lower_bound())
    return np.vstack(A), np.hstack(b)


def mpc_order(prog, u, x0):
    order = np.zeros(prog.num_vars())
    order[:] = np.inf
    for (i, var) in enumerate(itertools.chain(u.flat, x0.flatten(order='F'))):
        order[prog.FindDecisionVariableIndex(var)] = i
    return np.argsort(order)


def eliminate_equality_constrained_variables(C, d):
    """
    Given C and d defining a set of linear equality constraints:

        C x == d

    find a matrix W such that C x == d implies x = W z for some z \subset x

    This allows us to rewrite a QP with equality constraints into a QP over
    fewer variables with no equality constraints.
    """
    assert np.all(d == 0), "Right-hand side of the equality constraints must be zero"
    C = C.copy()
    num_vars = C.shape[1]
    W = np.eye(num_vars)
    for j in range(C.shape[1] - 1, C.shape[1] - C.shape[0] - 1, -1):
        nonzeros = np.nonzero(C[:, j])[0]
        if len(nonzeros) != 1:
            raise ValueError("C must be triangular (up to permutation). Try permuting the problem to mpc_order()")
        i = nonzeros[0]
        v = C[i, :j] / -C[i, j]
        W = W.dot(np.vstack([np.eye(j), v]))
        C = C[[k for k in range(C.shape[0]) if k != i], :]
    return W


def extract_objective(prog):
    num_vars = prog.num_vars()
    Q = np.zeros((num_vars, num_vars))
    q = np.zeros(num_vars)

    for binding in prog.linear_costs():
        var_order = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        ai = binding.constraint().A()
        assert ai.shape[0] == 1
        ai = ai[0, :]
        for i in range(ai.size):
            q[var_order[i]] += ai[i]

    for binding in prog.quadratic_costs():
        var_order = [prog.FindDecisionVariableIndex(v) for v in binding.variables()]
        Qi = binding.constraint().Q()
        bi = binding.constraint().b()
        for i in range(bi.size):
            q[var_order[i]] += bi[i]
            for j in range(bi.size):
                Q[var_order[i], var_order[j]] += Qi[i, j]
    Q = 0.5 * (Q + Q.T)
    return Q, q

def permutation_matrix(order):
    """
    Returns a matrix P such that P * y = y[order]
    """
    P = np.zeros((len(order), len(order)))
    for i in range(len(order)):
        P[i, order[i]] = 1

    return P


class SimpleQuadraticProgram(object):
    """
    Represents a quadratic program of the form:

    minimize 0.5 x' H x + f' x
       x
    such that A x <= b
              C x == d

    This class is meant to be used as a temporary container while performing
    variable substitutions and other transformations on the optimization
    problem.
    """
    def __init__(self, H, f, A, b, C=None, d=None):
        self.H = H
        self.f = f
        self.A = A
        self.b = b
        if C is None or d is None:
            assert C is None and d is None, "C and d must both be provided together"
            C = np.zeros((0, A.shape[1]))
            d = np.zeros(0)
        self.C = C
        self.d = d

    @property
    def num_vars(self):
        return self.A.shape[1]

    @staticmethod
    def from_mathematicalprogram(prog):
        """
        Construct a simple quadratic program representation from a symbolic
        MathematicalProgram. Note that this destroys the sparsity pattern of
        the MathematicalProgram's constraints and costs.
        """
        C, d = extract_linear_equalities(prog)
        A, b = extract_linear_inequalities(prog)
        Q, q = extract_objective(prog)
        return SimpleQuadraticProgram(Q, q, A, b, C, d)

    def to_mathematicalprogram(self):
        prog = mp.MathematicalProgram()
        x = prog.NewContinuousVariables(self.A.shape[1], "x")
        for i in range(self.A.shape[0]):
            prog.AddLinearConstraint(self.A[i, :].dot(x) <= self.b[i])
        for i in range(self.C.shape[0]):
            prog.AddLinearConstraint(self.C[i, :].dot(x) == self.d[i])
        prog.AddQuadraticCost(self.H, self.f, x)
        return prog, x

    def solve(self):
        prog, x = self.to_mathematicalprogram()
        prog.Solve()
        return prog.GetSolution(x)

    def affine_variable_substitution(self, T, u):
        """
        Given an optimization of the form:
        minimize 0.5 x' H x + f' x
           x
        such that A x <= b
                  C x == d

        perform a variable substitution defined by:
            x = T y + u

        and return an equivalent optimization over y.
        """

        # 0.5 (T y + u)' H (T y + u) + f' (T y + u)
        # 0.5 ( y' T' H T y + y' T' H u + u' H T y + u' H u) + f' T y + f' u
        # 0.5 y' T' H T y + u' H T y + f' T y + f' u + 0.5 u' H u
        # eliminate constants:
        # 0.5 y' T' H T y + (u' H T + f' T) y
        #
        # A x <= b
        # A (T y + u) <= b
        # A T y <= b - A u
        #
        # C x == d
        # C (T y + u) == d
        # C T y == d - C u

        H = T.T.dot(self.H).dot(T)
        f = u.dot(self.H).dot(T) + self.f.dot(T)
        A = self.A.dot(T)
        b = self.b - self.A.dot(u)
        C = self.C.dot(T)
        d = self.d - self.C.dot(u)
        return SimpleQuadraticProgram(H, f, A, b, C, d)

    def eliminate_equality_constrained_variables(self):
        """
        Given:
            - self: an optimization program over variables x
        Returns:
            - new_program: a new optimization program over variables
                           z \subset x with all equality-constrained variables
                           eliminated by solving C x == d
            - W: a matrix such that x = W z
        """

        W = eliminate_equality_constrained_variables(self.C, self.d)
        # x = W z
        new_program = self.affine_variable_substitution(W, np.zeros(self.num_vars))
        mask = np.ones(new_program.C.shape[0], dtype=np.bool)
        for i in range(new_program.C.shape[0]):
            if np.allclose(new_program.C[i, :], 0):
                assert np.isclose(new_program.d[i], 0)
                mask[i] = False
        new_program.C = new_program.C[mask, :]
        new_program.d = new_program.d[mask]
        return new_program, W

    def permute_variables(self, new_order):
        """
        Given:
            - self: an optimization program over variables x
            - new_order: a new ordering of variables
        Returns:
            - new_program: an optimization program over variables z such that
                           z = x[new_order]
            - P: a permutation matrix such that x = P z = P x[new_order]
        """
        assert len(new_order) == self.num_vars
        P = np.linalg.inv(permutation_matrix(new_order))
        # x = P x[new_order]
        new_program = self.affine_variable_substitution(P, np.zeros(self.num_vars))
        return new_program, P

class CanonicalMPCQP(object):
    """
    Represents a model-predictive control quadratic program of the form:

    minimize 0.5 u' H u + x' F u
      u, x
    such that G u <= W + E x
    """
    def __init__(self, H, F, G, W, E):
        self.H = H
        self.F = F
        self.G = G
        self.W = W
        self.E = E

    @staticmethod
    def from_mathematicalprogram(prog, u, x):
        u = np.asarray(u)
        x = np.asarray(x)
        qp = SimpleQuadraticProgram.from_mathematicalprogram(prog)
        order = mpc_order(prog, u, x)
        qp, P = qp.permute_variables(order)
        qp, W = qp.eliminate_equality_constrained_variables()

        assert np.allclose(qp.f, 0)
        assert np.allclose(qp.C, 0)
        assert np.allclose(qp.d, 0)

        nu = u.size
        H = qp.H[:nu, :nu]
        F = qp.H[nu:, :nu]
        G = qp.A[:, :nu]
        W = qp.b
        E = -qp.A[:, nu:]
        return CanonicalMPCQP(H, F, G, W, E)

def generate_mpc_system(prog, u, x0):
    C, d = extract_linear_equalities(prog)
    A, b = extract_linear_inequalities(prog)
    Q, q = extract_objective(prog)
    assert np.allclose(q, 0), "linear objective terms are not yet implemented"
    return Q, q, A, b, C, d

    order = mpc_order(prog, u, x0)
    P = permutation_matrix(order)
    # yhat = P * y
    # y = P^-1 * yhat

    Pinv = np.linalg.inv(P)

    Qhat = Pinv.T.dot(Q).dot(Pinv)
    qhat = q.dot(Pinv)
    Ahat = A.dot(Pinv)
    bhat = b
    Chat = C.dot(Pinv)
    dhat = d

    for i in range(10):
        f = rand(q.size)
        fhat = f.dot(Pinv)
        y, cost = mpc.quadratic_program(Q, f, A, b, C, d)
        yhat, cost_hat = mpc.quadratic_program(Qhat, fhat, Ahat, bhat, Chat, dhat)
        assert np.isclose(cost, cost_hat)
        assert np.allclose(yhat, P.dot(y))

    W = simplify(Chat)
    # yhat = W * z

    Qtilde = W.T.dot(Qhat).dot(W)
    qtilde = qhat.dot(W)
    Atilde = Ahat.dot(W)
    btilde = bhat
    Ctilde = Chat.dot(W)
    dtilde = dhat
    assert np.allclose(Ctilde, 0)

    for i in range(100):
        f = rand(q.size)
        fhat = f.dot(Pinv)
        ftilde = fhat.dot(W)
        y, cost = mpc.quadratic_program(Q, f, A, b, C, d)
        yhat, cost_hat = mpc.quadratic_program(Qhat, fhat, Ahat, bhat, Chat, dhat)
        ytilde, cost_tilde = mpc.quadratic_program(Qtilde, ftilde, Atilde, btilde)
        assert np.isclose(cost, cost_hat)
        assert np.isclose(cost, cost_tilde)
        assert np.allclose(yhat, P.dot(y))
        assert np.allclose(yhat, W.dot(ytilde))

    H = Qtilde[:u.size, :u.size]
    F = Qtilde[u.size:, :u.size]
    G = Atilde[:, :u.size]
    E = -Atilde[:, u.size:]
    W = btilde

    return H, F, G, W, E