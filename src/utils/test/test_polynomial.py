import unittest
from utils.polynomial import Polynomial


class TestPolynomial(unittest.TestCase):
    def test_polynomial_eval(self):
        p = Polynomial([1, 2, 3, 4])
        for x in range(-5, 5):
            self.assertAlmostEqual(p(x), 1 + 2 * x + 3 * x**2 + 4 * x**3)

    def test_polynomial_degree(self):
        p = Polynomial([1, 2, 3, 4])
        self.assertEqual(p.degree, 3)

    def test_polynomial_derivative(self):
        p = Polynomial([3, 1, 7, 8, 2])
        dp = p.derivative()
        self.assertEqual(dp.coeffs, [1, 2 * 7, 3 * 8, 4 * 2])

    def test_polynomial_map(self):
        p = Polynomial([1, 2, 3, 4])
        p2 = p.map(lambda x: x + 1)
        self.assertEqual(p2.coeffs, [2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
