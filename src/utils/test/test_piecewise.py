import unittest
from utils.piecewise import Piecewise


class TestPiecewise(unittest.TestCase):
    def test_from_above(self):
        p = Piecewise([1, 2, 3],
                      [lambda x: 1,
                       lambda x: 2])
        self.assertEqual(p.from_above(1), 1)
        self.assertEqual(p.from_above(1.5), 1)
        self.assertEqual(p.from_above(2), 2)
        self.assertEqual(p.from_above(2.5), 2)
        self.assertEqual(p(1), 1)
        self.assertEqual(p(1.5), 1)
        self.assertEqual(p(2), 2)
        self.assertEqual(p(2.5), 2)

    def test_from_below(self):
        p = Piecewise([1, 2, 3],
                      [lambda x: 1,
                       lambda x: 2])
        self.assertEqual(p.from_below(1.5), 1)
        self.assertEqual(p.from_below(2), 1)
        self.assertEqual(p.from_below(2.5), 2)
        self.assertEqual(p.from_below(3), 2)
        self.assertEqual(list(p.at_all_breaks()), [1, 2, 2])

    def test_map(self):
        p = Piecewise([1, 2, 3],
                      [lambda x: 1,
                       lambda x: 2])
        self.assertEqual(p(1), 1)
        self.assertEqual(p(1.5), 1)
        self.assertEqual(p(2), 2)
        self.assertEqual(p(2.5), 2)

        p2 = p.map(lambda f: lambda x: 2 * f(x))
        self.assertEqual(p2(1), 2)
        self.assertEqual(p2(1.5), 2)
        self.assertEqual(p2(2), 4)
        self.assertEqual(p2(2.5), 4)


if __name__ == '__main__':
    unittest.main()
