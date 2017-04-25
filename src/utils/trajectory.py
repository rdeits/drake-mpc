from __future__ import absolute_import, division, print_function

class Trajectory(object):
    def __init__(self, components, constructor):
        self.components = components
        self.constructor = constructor

    def __call__(self, t):
        args = []
        for c in self.components:
            if callable(c):
                args.append(c(t))
            else:
                # assume c is a list of callable things
                args.append([c_i(t) for c_i in c])

        return self.constructor(*args)
