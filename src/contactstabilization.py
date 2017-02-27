from __future__ import absolute_import, division, print_function

from itertools import islice, chain
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from polynomial import Polynomial
from piecewise import Piecewise
from trajectory import Trajectory
from boxatlas import BoxAtlasState, BoxAtlasInput

def add_continuity_constraints(prog, piecewise):
    for t in piecewise.breaks[1:-1]:
        frombelow = piecewise.from_below(t)
        fromabove = piecewise.from_above(t)
        for i in range(frombelow.size):
            prog.AddLinearConstraint(frombelow.flat[i] == fromabove.flat[i])


def piecewise_polynomial_variable(prog, domain, dimension, degree, kind="continuous"):
    if kind == "continuous":
        C = [prog.NewContinuousVariables(degree + 1, len(domain) - 1) for _ in range(dimension)]
    elif kind == "binary":
        C = [prog.NewBinaryVariables(degree + 1, len(domain) - 1) for _ in range(dimension)]
    else:
        raise ValueError("Expected kind to be 'continuous' or 'binary', but got {:s}".format(kind))
    C = np.stack(C, axis=2)
    assert C.shape == (degree + 1, len(domain) - 1, dimension)
    return Piecewise(domain,
                     [Polynomial([C[i, j, :] for i in range(degree + 1)])
                          for j in range(len(domain) - 1)])


def add_limb_velocity_constraints(prog, qcom, qlimb, vmax, dt):
    ts = qcom.breaks
    dim = qcom(ts[0]).size
    vcom = qcom.derivative()
    for j in range(len(ts) - 2):
        relative_velocity = 1.0 / dt * (qlimb(ts[j + 1]) - qlimb(ts[j])) - vcom(ts[j])
        for i in range(dim):
            prog.AddLinearConstraint((relative_velocity[i] - vmax) <= 0)
            prog.AddLinearConstraint((-relative_velocity[i] - vmax) <= 0)


def add_dynamics_constraints(prog, robot, qcom, contact_force):
    dim = robot.dim
    num_limbs = len(robot.limb_bounds)
    gravity_force = np.zeros(dim)
    gravity_force[-1] = -robot.mass * robot.g
    ts = qcom.breaks
    acom = qcom.derivative().derivative()
    for t in ts[:-1]:
        for i in range(dim):
            total_contact_force = sum([contact_force[k](t)[i] for k in range(num_limbs)])
            prog.AddLinearConstraint(total_contact_force + gravity_force[i] == robot.mass * acom(t)[i])


def add_no_force_at_distance_constraints(prog, contact, contact_force, Mbig):
    ts = contact.breaks
    dim = contact_force(ts[0]).size
    for t in ts[:-1]:
        for i in range(dim):
            prog.AddLinearConstraint((contact_force(t)[i] - (Mbig * contact(t)[0])) <= 0)
            prog.AddLinearConstraint((-contact_force(t)[i] - (Mbig * contact(t)[0])) <= 0)


def add_contact_surface_constraints(prog, qlimb, surface, contact, Mbig):
    ts = qlimb.breaks
    for j in range(len(ts) - 1):
        t = ts[j]
        A = surface.pose_constraints.getA()
        b = surface.pose_constraints.getB()
        qlimb_after_dt = qlimb.from_below(ts[j + 1])
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(qlimb_after_dt) - (b[i] + Mbig * (1 - contact(t)[0]))) <= 0)


def add_contact_force_constraints(prog, contact_force, surface, contact, Mbig):
    ts = contact_force.breaks
    for t in ts[:-1]:
        A = surface.force_constraints.getA()
        b = surface.force_constraints.getB()
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(contact_force(t)) - (b[i] + Mbig * (1 - contact(t)[0]))) <= 0)

def add_contact_velocity_constraints(prog, qlimb, contact, Mbig):
    ts = qlimb.breaks
    dim = qlimb(0).size

    vlimb = qlimb.derivative()
    for j in range(len(ts) - 2):
        t = ts[j]
        tnext = ts[j + 1]
        indicator = contact(t)[0]
        for i in range(dim):
            prog.AddLinearConstraint((vlimb(tnext)[i] - (Mbig * (1 - indicator))) <= 0)
            prog.AddLinearConstraint((-vlimb(tnext)[i] - (Mbig * (1 - indicator))) <= 0)
    # for j in range(len(ts) - 1):
    #     t = ts[j]
    #     indicator = contact(t)[0]
    #     if j < len(ts) - 3:
    #         for i in range(dim):
    #             prog.AddLinearConstraint(((qlimb(ts[j + 1])[i] - qlimb(ts[j + 2])[i]) - (Mbig * (1 - indicator))) <= 0)
    #             prog.AddLinearConstraint((-(qlimb(ts[j + 1])[i] - qlimb(ts[j + 2])[i]) - (Mbig * (1 - indicator))) <= 0)


def get_piecewise_solution(prog, piecewise):
    return piecewise.map(lambda p: p.map(lambda x: prog.GetSolution(x)))


def extract_solution(prog, robot, qcom, qlimb, contact, contact_force):
    qcom = get_piecewise_solution(prog, qcom)
    vcom = qcom.map(Polynomial.derivative)
    qlimb = [get_piecewise_solution(prog, q) for q in qlimb]
    flimb = [get_piecewise_solution(prog, f) for f in contact_force]
    contact = [get_piecewise_solution(prog, c) for c in contact]
    return (Trajectory(
        [qcom, vcom] + qlimb,
        lambda qcom, vcom, *qlimb: BoxAtlasState(robot, qcom=qcom, vcom=vcom, qlimb=qlimb)
    ), Trajectory(
        flimb,
        lambda *flimb: BoxAtlasInput(robot, flimb=flimb)
    ), contact)


def add_kinematic_constraints(prog, qlimb, qcom, polytope):
    A = polytope.getA()
    b = polytope.getB()
    for (ql, qc) in islice(zip(qlimb.at_all_breaks(), qcom.at_all_breaks()), 1, None):
        offset = ql - qc
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(offset) - b[i]) <= 0)


def count_contact_switches(prog, contact):
    ts = contact.breaks
    delta = prog.NewContinuousVariables(len(ts) - 2, "contact_delta")
    for j in range(len(ts) - 2):
        prog.AddLinearConstraint(contact(ts[j + 1])[0] - contact(ts[j])[0] <= delta[j])
        prog.AddLinearConstraint(-(contact(ts[j + 1])[0] - contact(ts[j])[0]) <= delta[j])
    total_switches = prog.NewContinuousVariables(1, "contact_switches")[0]
    prog.AddLinearConstraint(sum(delta) <= total_switches)
    return total_switches


def contact_stabilize(initial_state, env):
    robot = initial_state.robot
    dt = 0.05
    time_horizon = 10 * dt
    ts = np.linspace(0, time_horizon, time_horizon / dt + 1)
    dim = robot.dim
    num_limbs = len(robot.limb_bounds)
    vlimb_max = 5

    Mq = 10
    Mv = 100
    Mf = 1000

    prog = mp.MathematicalProgram()
    qcom = piecewise_polynomial_variable(prog, ts, dim, 2)
    add_continuity_constraints(prog, qcom)
    vcom = qcom.derivative()
    add_continuity_constraints(prog, vcom)

    qlimb = [piecewise_polynomial_variable(prog, ts, dim, 1) for k in range(num_limbs)]
    vlimb = [q.derivative() for q in qlimb]
    for q in qlimb:
        add_continuity_constraints(prog, q)
    contact_force = [piecewise_polynomial_variable(prog, ts, dim, 0) for k in range(num_limbs)]
    contact = [piecewise_polynomial_variable(prog, ts, 1, 0, kind="binary") for k in range(num_limbs)]

    for k in range(num_limbs):
        add_no_force_at_distance_constraints(prog, contact[k], contact_force[k], Mf)
        add_contact_surface_constraints(prog, qlimb[k], env.surfaces[k], contact[k], Mq)
        add_contact_force_constraints(prog, contact_force[k], env.surfaces[k], contact[k], Mf)
        add_contact_velocity_constraints(prog, qlimb[k], contact[k], Mv)
        add_limb_velocity_constraints(prog, qcom, qlimb[k], vlimb_max, dt)
        add_kinematic_constraints(prog, qlimb[k], qcom, robot.limb_bounds[k])
        switches = count_contact_switches(prog, contact[k])
        prog.AddLinearConstraint(switches <= 1)

    A = env.free_space.getA()
    b = env.free_space.getB()
    for q in chain(qcom.at_all_breaks(), *[ql.at_all_breaks() for ql in qlimb]):
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(q) - b[i]) <= 0)


    add_dynamics_constraints(prog, robot, qcom, contact_force)


    for i in range(dim):
        prog.AddLinearConstraint(qcom(ts[0])[i] == initial_state.qcom[i])
        prog.AddLinearConstraint(vcom(ts[0])[i] == initial_state.vcom[i])
        for k in range(num_limbs):
            prog.AddLinearConstraint(vlimb[k](ts[0])[i] == 0)
            prog.AddLinearConstraint(qlimb[k](ts[0])[i] == initial_state.qlimb[k][i])

    prog.AddQuadraticCost(0.001 * np.sum(np.sum(np.power(contact_force[k](t), 2)) for t in ts[:-1] for k in range(num_limbs)))
    prog.AddQuadraticCost(100 * np.sum(np.sum(np.power(q - np.array([0, 1]), 2)) for q in qcom.at_all_breaks()))
    prog.AddQuadraticCost(100 * np.sum(np.power(qcom.from_below(ts[-1]) - np.array([0, 1]), 2)))
    prog.AddQuadraticCost(100 * np.sum(10 * np.power(vcom.from_below(ts[-1]) - np.array([0, 0]), 2)))

    qcomf = qcom.from_below(ts[-1])
    qlimbf = [qlimb[k].from_below(ts[-1]) for k in range(num_limbs)]
    prog.AddQuadraticCost(1000 * (qlimbf[1][0] - (qcomf[0] + 0.25))**2)
    prog.AddQuadraticCost(1000 * (qlimbf[2][0] - (qcomf[0] - 0.25))**2)


    result = prog.Solve()
    print(result)
    assert result == mp.SolutionResult.kSolutionFound

    return extract_solution(prog, robot, qcom, qlimb, contact, contact_force)

