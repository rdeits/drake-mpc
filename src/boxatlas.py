from __future__ import absolute_import, division, print_function

from collections import namedtuple

from director import viewerclient as vc
from irispy import Polyhedron
import numpy as np


class BoxAtlas(object):
    dim = 2
    g = 9.81
    mass = 10
    limb_velocity_limits = [
        10,
        10,
        10,
        10
    ]

    limb_bounds = [
        Polyhedron.fromBounds([0.5, -0.5], [1.0, 0.5]),    # right arm
        Polyhedron.fromBounds([0.0, -1.0], [0.5, -0.5]),   # right leg
        Polyhedron.fromBounds([-0.5, -1.0], [0.0, -0.5]),  # left leg
        Polyhedron.fromBounds([-1.0, -0.5], [-0.5, 0.5])  # left arm
    ]

    def __init__(self):
        pass


def draw(vis, state, atlasinput):
    state.draw(vis)
    atlasinput.draw(vis)

class BoxAtlasState(object):
    def __init__(self, robot, qcom=None, vcom=None, qlimb=None):
        self.robot = robot
        if qcom is None:
            qcom = np.zeros(robot.dim)
        if vcom is None:
            vcom = np.zeros(robot.dim)
        if qlimb is None:
            qlimb = [np.zeros(robot.dim) for _ in robot.limb_bounds]

        self.qcom = qcom
        self.vcom = vcom
        self.qlimb = qlimb

    def draw(self, vis):
        vis["body"].setgeometry(vc.Box(lengths=[0.1, 0.1, 0.1]))
        vis["body"].settransform(vc.transformations.translation_matrix([self.qcom[0], 0, self.qcom[1]]))
        for (i, q) in enumerate(self.qlimb):
            v = vis["limb_{:d}".format(i)]
            v.setgeometry(vc.Sphere(radius=0.05))
            v.settransform(vc.transformations.translation_matrix([q[0], 0, q[1]]))


class BoxAtlasInput(object):
    def __init__(self, robot, flimb=None):
        self.robot = robot
        if flimb is None:
            flimb = [np.zeros(robot.dim) for _ in robot.limb_bounds]
        self.flimb = flimb

    def draw(self, vis):
        for (i, q) in enumerate(self.qlimb):
            v = vis["limb_{:d}_force".format(i)]
            v.setgeometry(vc.Sphere(radius=0.05))
            v.settransform(vc.transformations.translation_matrix([q[0], 0, q[1]]))


Surface = namedtuple("Surface", ["pose_constraints", "force_constraints"])

class Environment(object):
    def __init__(self, surfaces):
        self.surfaces = surfaces