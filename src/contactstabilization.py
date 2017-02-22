from __future__ import absolute_import, division, print_function

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
            prog.AddLinearConstraint(frombelow.flat[i].Expand() == fromabove.flat[i].Expand())


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


def add_velocity_constraints(prog, q, vmax, dt):
    ts = q.breaks
    dim = q(ts[0]).size
    for j in range(len(ts) - 2):
        for i in range(dim):
            prog.AddLinearConstraint((q(ts[j + 1])[i] - q(ts[j])[i] - vmax * dt).Expand() <= 0)
            prog.AddLinearConstraint((-1 * (q(ts[j + 1])[i] - q(ts[j])[i]) - vmax * dt).Expand() <= 0)


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
            prog.AddLinearConstraint((contact_force(t)[i] - (Mbig * contact(t)[0])).Expand() <= 0)


def add_contact_surface_constraints(prog, qlimb, surface, contact, Mbig):
    ts = qlimb.breaks
    for j in range(len(ts) - 1):
        t = ts[j]
        A = surface.pose_constraints.getA()
        b = surface.pose_constraints.getB()
        qlimb_after_dt = qlimb.from_below(ts[j + 1])
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(qlimb_after_dt) - (b[i] + Mbig * (1 - contact(t)[0]))).Expand() <= 0)


def add_contact_force_constraints(prog, contact_force, surface, contact, Mbig):
    ts = contact_force.breaks
    for t in ts[:-1]:
        A = surface.force_constraints.getA()
        b = surface.force_constraints.getB()
        for i in range(A.shape[0]):
            prog.AddLinearConstraint((A[i, :].dot(contact_force(t)) - (b[i] + Mbig * (1 - contact(t)[0]))).Expand() <= 0)


def get_piecewise_solution(prog, piecewise):
    return piecewise.map(lambda p: p.map(lambda x: prog.GetSolution(x)))


def contact_stabilize(initial_state, env):
    robot = initial_state.robot
    dt = 0.1
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
    acom = qcom.derivative()

    qlimb = [piecewise_polynomial_variable(prog, ts, dim, 0) for k in range(num_limbs)]
    contact_force = [piecewise_polynomial_variable(prog, ts, dim, 0) for k in range(num_limbs)]
    contact = [piecewise_polynomial_variable(prog, ts, 1, 0, kind="binary") for k in range(num_limbs)]

    for k in range(num_limbs):
        add_no_force_at_distance_constraints(prog, contact[k], contact_force[k], Mf)
        add_contact_surface_constraints(prog, qlimb[k], env.surfaces[k], contact[k], Mq)
        add_contact_force_constraints(prog, contact_force[k], env.surfaces[k], contact[k], Mf)

        for j in range(len(ts) - 1):
            t = ts[j]
            indicator = contact[k](t)[0]
            if j < len(ts) - 3:
                for i in range(dim):
                    prog.AddLinearConstraint(((qlimb[k](ts[j + 1])[i] - qlimb[k](ts[j + 2])[i]) - (Mv * (1 - indicator))).Expand() <= 0)
                    prog.AddLinearConstraint((-1 * (qlimb[k](ts[j + 1])[i] - qlimb[k](ts[j + 2])[i]) - (Mv * (1 - indicator))).Expand() <= 0)

    for k in range(num_limbs):
        add_velocity_constraints(prog, qlimb[k], vlimb_max, dt)

    # TODO: hard-coded free space
    for t in ts[:-1]:
        for i in range(dim):
            prog.AddLinearConstraint(qcom(t)[i] <= 1)
            prog.AddLinearConstraint(qcom(t)[i] >= -1)

            for k in range(num_limbs):
                prog.AddLinearConstraint(qlimb[k](t)[i] <= 1)
                prog.AddLinearConstraint(qlimb[k](t)[i] >= -1)

    add_dynamics_constraints(prog, robot, qcom, contact_force)

    for i in range(dim):
        prog.AddLinearConstraint(qcom(ts[0])[i] == initial_state.qcom[i])
        prog.AddLinearConstraint(vcom(ts[0])[i].Expand() == initial_state.vcom[i])
        for k in range(num_limbs):
            prog.AddLinearConstraint(qlimb[k](ts[0])[i] == initial_state.qlimb[k][i])

    for t in ts[1:]:
        for (k, polytope) in enumerate(robot.limb_bounds):
            A = polytope.getA()
            b = polytope.getB()
            offset = qlimb[k].from_below(t) - qcom.from_below(t)
            for i in range(A.shape[0]):
                prog.AddLinearConstraint((A[i, :].dot(offset) - b[i]).Expand() <= 0)

    all_force_vars = np.hstack([contact_force[k](t) for t in ts[:-1] for k in range(num_limbs)])
    print(all_force_vars.shape)
    prog.AddQuadraticCost(0.01 * np.eye(len(all_force_vars)),
                          np.zeros(len(all_force_vars)),
                          all_force_vars)
    all_qcom_vars = np.hstack([p[0] for p in qcom.functions])
    print(all_qcom_vars.shape)
    prog.AddQuadraticCost(0.01 * np.eye(len(all_qcom_vars)),
                          np.zeros(len(all_qcom_vars)),
                          all_qcom_vars)


    result = prog.Solve()
    print(result)
    assert result == mp.SolutionResult.kSolutionFound

    qcom = get_piecewise_solution(prog, qcom)
    vcom = qcom.map(Polynomial.derivative)
    qlimb = [get_piecewise_solution(prog, qlimb[k]) for k in range(num_limbs)]
    flimb = [get_piecewise_solution(prog, contact_force[k]) for k in range(num_limbs)]
    contact = [get_piecewise_solution(prog, contact[k]) for k in range(num_limbs)]
    return (Trajectory(
        [qcom, vcom] + qlimb,
        lambda qcom, vcom, *qlimb: BoxAtlasState(robot, qcom=qcom, vcom=vcom, qlimb=qlimb)
    ), Trajectory(
        flimb,
        lambda *flimb: BoxAtlasInput(robot, flimb=flimb)
    ), contact)