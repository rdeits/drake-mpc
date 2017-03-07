import unittest
import numpy as np
import mpc.symbolic as sym
from mpc.mpc_tools import quadratic_program
import pydrake.solvers.mathematicalprogram as mp


class TestQuadraticProgram(unittest.TestCase):
    def test_round_trip(self):

        np.random.seed(0)
        for i in range(20):
            x_goal = np.random.rand(3) * 20 - 10
            prog1 = mp.MathematicalProgram()
            x = prog1.NewContinuousVariables(3, "x")
            prog1.AddLinearConstraint(x[0] >= 1)
            prog1.AddLinearConstraint(x[0] <= 10)
            prog1.AddLinearConstraint(x[0] + 5 * x[1] <= 11)
            prog1.AddLinearConstraint(-x[1] + 5 * x[0] <= 5)
            prog1.AddLinearConstraint(x[2] == x[0] + -2 * x[1])
            prog1.AddQuadraticCost(np.sum(np.power(x - x_goal, 2)))
            prog1.Solve()
            xstar_prog1 = prog1.GetSolution(x)
            simple = sym.SimpleQuadraticProgram.from_mathematicalprogram(prog1)
            xstar_qp, cost = quadratic_program(simple.H, simple.f,
                                           simple.A, simple.b,
                                           simple.C, simple.d)
            self.assertTrue(np.allclose(xstar_prog1, xstar_qp.flatten(), atol=1e-7))

            prog2, x2 = simple.to_mathematicalprogram()
            prog2.Solve()
            xstar_prog2 = prog2.GetSolution(x2)
            self.assertTrue(np.allclose(xstar_prog1, xstar_prog2))

    def test_substitution(self):
        np.random.seed(1)
        for i in range(20):
            x_goal = np.random.rand(2) * 10 - 5
            prog = mp.MathematicalProgram()
            x = prog.NewContinuousVariables(2, "x")
            prog.AddLinearConstraint(x[0] >= 1)
            prog.AddLinearConstraint(x[1] >= 1)
            prog.AddLinearConstraint(x[0] + x[1] <= np.random.randint(4, 10))
            prog.AddQuadraticCost(np.sum(np.power(x - x_goal, 2)))

            simple_x = sym.SimpleQuadraticProgram.from_mathematicalprogram(prog)
            # x = T y + u
            T = np.random.rand(2, 2) - 0.5
            u = np.random.rand(2) - 0.5
            simple_y = simple_x.affine_variable_substitution(T, u)

            ystar = simple_y.solve()
            xstar = simple_x.solve()

            self.assertTrue(np.allclose(T.dot(ystar) + u, xstar))

    def test_permutation(self):
        np.random.seed(2)
        for i in range(20):
            x_goal = np.random.rand(3) * 20 - 10
            prog = mp.MathematicalProgram()
            x = prog.NewContinuousVariables(3, "x")
            prog.AddLinearConstraint(x[0] >= 1)
            prog.AddLinearConstraint(x[0] <= 10)
            prog.AddLinearConstraint(x[0] + 5 * x[1] <= 11)
            prog.AddLinearConstraint(-x[1] + 5 * x[0] <= 5)
            prog.AddLinearConstraint(x[2] == x[0] + -2 * x[1])
            prog.AddQuadraticCost(np.sum(np.power(x - x_goal, 2)))

            order = np.arange(x.size)
            np.random.shuffle(order)

            simple_x = sym.SimpleQuadraticProgram.from_mathematicalprogram(prog)

            # y = P * x = x[order]
            P = sym.permutation_matrix(order)
            # x = P^-1 * y
            simple_y = simple_x.affine_variable_substitution(np.linalg.inv(P),
                                                             np.zeros(x.size))

            ystar = simple_y.solve()
            xstar = simple_x.solve()

            self.assertTrue(np.allclose(ystar, xstar[order]))


if __name__ == '__main__':
    unittest.main()
