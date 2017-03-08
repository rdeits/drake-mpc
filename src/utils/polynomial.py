from __future__ import absolute_import, division, print_function
import unittest


class Polynomial(object):
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __call__(self, x):
        return self.evaluate(x)

    @property
    def degree(self):
        return len(self.coeffs) - 1

    def evaluate(self, x):
        y = self.coeffs[-1]
        for i in range(self.degree - 1, -1, -1):
            y = self[i] + x * y
        return y

    def derivative(self):
        return Polynomial([i * self[i] for i in range(1, self.degree + 1)])

    def map(self, function):
        return Polynomial(map(function, self.coeffs))
