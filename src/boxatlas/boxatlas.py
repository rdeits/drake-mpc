from __future__ import absolute_import, division, print_function

from collections import namedtuple

from director import viewerclient as vc
from irispy import Polyhedron
import numpy as np
import time
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


class BoxAtlas(object):
    dim = 2
    g = 9.81
    mass = 10
    limb_velocity_limits = [
        10,
        20,
        20,
        10
    ]

    limb_bounds = [
        Polyhedron.fromBounds([0.25, -0.7], [0.75, 0.3]),    # right arm
        Polyhedron.fromBounds([0.05, -1.12], [0.35, -0.5]),   # right leg
        Polyhedron.fromBounds([-0.35, -1.12], [-0.05, -0.5]),  # left leg
        Polyhedron.fromBounds([-0.75, -0.7], [-0.25, 0.3])  # left arm
    ]

    limb_idx_map = {"right_arm": 0, "right_leg": 1, "left_leg":2,
                    "left_arm":3}

    def __init__(self):
        pass


def draw(vis, state, atlasinput=None, env=None):
    vis["body"].setgeometry(vc.GeometryData(vc.Box(lengths=[0.25, 0.1, .5]), color=[0.8, 0.4, 0.4, 0.8]))
    vis["body"].settransform(vc.transformations.translation_matrix([state.qcom[0], 0, state.qcom[1]]))
    for (i, q) in enumerate(state.qlimb):
        limb_vis = vis["limb_{:d}".format(i)]
        origin = np.array([q[0], 0, q[1]])
        limb_vis.settransform(vc.transformations.translation_matrix(origin))

        v = limb_vis["position"]
        v.setgeometry(vc.Sphere(radius=0.05))

        if atlasinput is not None:
            v = limb_vis["force"]
            force = np.array([atlasinput.flimb[i][0], 0, atlasinput.flimb[i][1]])
            v.setgeometry(vc.PolyLine(points=[[0, 0, 0], list(0.005 * force)], end_head=True))

    if env is not None:
        for (i, surface) in enumerate(env.surfaces):
            verts2d = surface.pose_constraints.generatorPoints()
            assert len(verts2d) == 2
            length = np.linalg.norm(verts2d[1] - verts2d[0])
            origin = 0.5 * (verts2d[0] + verts2d[1])
            origin = [origin[0], 0, origin[1]]
            box = vc.Box(lengths=[length, length, 0.01])
            v = vis["environment"]["surface_{:d}".format(i)]
            v.setgeometry(box)
            angle = np.arctan2(verts2d[1][1] - verts2d[0][1], verts2d[1][0] - verts2d[0][0])
            v.settransform(vc.transformations.rotation_matrix(angle, [0, 1, 0], origin).dot(vc.transformations.translation_matrix(origin)))


def drawSinglePlanFrame(vis, solnData, t):
    states = solnData.states
    inputs = solnData.inputs
    ts = solnData.ts
    env = solnData.opt.env
    draw(vis, states(t), inputs(t), env)


def planPlayback(vis, solnData):
    # unpack solution
    states = solnData.states
    inputs = solnData.inputs
    ts = solnData.ts

    # draw solution plan
    for t in np.linspace(0, ts[-1] - 0.001, ts[-1] / 0.01):
        drawSinglePlanFrame(vis, solnData, t)
        time.sleep(0.05)


def planPlayback(vis, solnData, slider=False):
    """
    :param vis:
    :param solnData: namedtuple with fields
    opt - BoxAtlasContactStabilization
    states
    inputs
    contact_indicator
    ts
    :param: slider - whether or not to use slider
    :return: None
    """

    # unpack solution
    ts = solnData.ts

    if slider:
        slider = widgets.FloatSlider(min=ts[0], max=ts[-1] - 0.001, step=0.01, value=0)
        interact(drawSinglePlanFrame, vis=fixed(vis), solnData=fixed(solnData), t=slider)
    else:
        # draw solution plan
        for t in np.linspace(0, ts[-1] - 0.001, ts[-1] / 0.01):
            drawSinglePlanFrame(vis, solnData, t)
            time.sleep(0.05)


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

    def copy(self):
        return BoxAtlasState(robot=self.robot,
                             qcom=self.qcom.copy(),
                             vcom=self.vcom.copy(),
                             qlimb=[q.copy() for q in self.qlimb])


class BoxAtlasInput(object):
    def __init__(self, robot, flimb=None):
        self.robot = robot
        if flimb is None:
            flimb = [np.zeros(robot.dim) for _ in robot.limb_bounds]
        self.flimb = flimb


Surface = namedtuple("Surface", ["pose_constraints", "force_constraints"])

class Environment(object):
    def __init__(self, surfaces, free_space):
        self.surfaces = surfaces
        self.free_space = free_space
