from __future__ import absolute_import, division, print_function

class Trajectory(object):
    def __init__(self, components, constructor):
        self.components = components
        self.constructor = constructor

    def __call__(self, t):
        return self.constructor(*[c(t) for c in self.components])
