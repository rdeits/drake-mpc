from __future__ import absolute_import, division, print_function

import unittest
import numpy as np


class Piecewise(object):
    __slots__ = ["breaks", "functions"]

    def __init__(self, breaks, functions):
        self.breaks = np.asarray(breaks)
        assert all(breaks[i] <= breaks[i + 1] for i in range(len(breaks) - 1))
        assert len(breaks) == len(functions) + 1
        self.functions = functions

    def from_above(self, x):
        i = np.searchsorted(self.breaks, x, side="right")
        if i <= 0 or i >= len(self.breaks):
            raise ValueError("Input value {:g} is out of the allowable range [{:g}, {:g})".format(x, self.breaks[0], self.breaks[-1]))
        return self.functions[i - 1](x - self.breaks[i - 1])

    def from_below(self, x):
        i = np.searchsorted(self.breaks, x, side="left")
        if i <= 0 or i >= len(self.breaks):
            raise ValueError(
                "Input value {:g} is out of the allowable range ({:g}, {:g}]".format(x, self.breaks[0], self.breaks[-1]))
        return self.functions[i - 1](x - self.breaks[i - 1])

    def __call__(self, x):
        return self.from_above(x)

    def map(self, function, *args, **kwargs):
        return Piecewise(self.breaks, map(lambda f: function(f, *args, **kwargs), self.functions))

    def derivative(self, *args, **kwargs):
        return self.map(lambda f: f.derivative(*args, **kwargs))


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
