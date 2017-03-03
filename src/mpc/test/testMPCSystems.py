import unittest
import numpy as np

# https://docs.python.org/2/library/unittest.html

class TestMPCSystems(unittest.TestCase):
    def test_loading_from_A_and_B_matrices(self):
        A = np.zeros((2, 2))
        B = np.random.rand(2)
        self.assertEqual(A[0, 0], 0)


if __name__ == '__main__':
    unittest.main()
