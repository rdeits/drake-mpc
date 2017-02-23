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
    vis["body"].setgeometry(vc.Box(lengths=[0.1, 0.1, 0.1]))
    vis["body"].settransform(vc.transformations.translation_matrix([state.qcom[0], 0, state.qcom[1]]))
    for (i, q) in enumerate(state.qlimb):
        limb_vis = vis["limb_{:d}".format(i)]
        origin = np.array([q[0], 0, q[1]])
        limb_vis.settransform(vc.transformations.translation_matrix(origin))

        v = limb_vis["position"]
        v.setgeometry(vc.Sphere(radius=0.05))

        v = limb_vis["force"]
        force = np.array([atlasinput.flimb[i][0], 0, atlasinput.flimb[i][1]])
        v.setgeometry(vc.PolyLine(points=[[0, 0, 0], list(0.01 * force)], end_head=True))


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


class BoxAtlasInput(object):
    def __init__(self, robot, flimb=None):
        self.robot = robot
        if flimb is None:
            flimb = [np.zeros(robot.dim) for _ in robot.limb_bounds]
        self.flimb = flimb


Surface = namedtuple("Surface", ["pose_constraints", "force_constraints"])

class Environment(object):
    def __init__(self, surfaces):
        self.surfaces = surfaces