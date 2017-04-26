from __future__ import absolute_import, division, print_function

from itertools import islice, chain
from collections import namedtuple
import time
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.gurobi import GurobiSolver
from utils.polynomial import Polynomial
from utils.piecewise import Piecewise
from utils.trajectory import Trajectory
from boxatlas.boxatlas import BoxAtlasState, BoxAtlasInput


SolutionData = namedtuple("SolutionData",
                          ["opt" ,"states", "inputs", "contact_indicator", "ts", "solve_time"])

ContactVariables = namedtuple("ContactVariables", ["contact_lambda", "contact",
                                                   "contact_sequence_array"])


class MixedIntegerTrajectoryOptimization(mp.MathematicalProgram):
    def add_continuity_constraints(self, piecewise):
        for t in piecewise.breaks[1:-1]:
            frombelow = piecewise.from_below(t)
            fromabove = piecewise.from_above(t)
            for i in range(frombelow.size):
                self.AddLinearConstraint(frombelow.flat[i] == fromabove.flat[i])

    def new_piecewise_polynomial_variable(self, domain, dimension, degree, kind="continuous"):
        if kind == "continuous":
            C = [self.NewContinuousVariables(degree + 1, len(domain) - 1) for _ in range(dimension)]
        elif kind == "binary":
            C = [self.NewBinaryVariables(degree + 1, len(domain) - 1) for _ in range(dimension)]
        else:
            raise ValueError("Expected kind to be 'continuous' or 'binary', but got {:s}".format(kind))
        C = np.stack(C, axis=2)
        assert C.shape == (degree + 1, len(domain) - 1, dimension)
        return Piecewise(domain,
                         [Polynomial([C[i, j, :] for i in range(degree + 1)])
                              for j in range(len(domain) - 1)])

    def add_limb_velocity_constraints(self, qcom, qlimb, vmax, dt):
        ts = qcom.breaks
        dim = qcom(ts[0]).size
        vcom = qcom.derivative()
        for j in range(len(ts) - 2):
            relative_velocity = 1.0 / dt * (qlimb(ts[j + 1]) - qlimb(ts[j])) - vcom(ts[j])
            for i in range(dim):
                self.AddLinearConstraint((relative_velocity[i] - vmax) <= 0)
                self.AddLinearConstraint((-relative_velocity[i] - vmax) <= 0)

    def add_dynamics_constraints(self, robot, qcom, contact_force):
        dim = robot.dim
        num_limbs = len(robot.limb_bounds)
        gravity_force = np.zeros(dim)
        gravity_force[-1] = -robot.mass * robot.g
        ts = qcom.breaks
        acom = qcom.derivative().derivative()
        for t in ts[:-1]:
            for i in range(dim):
                total_contact_force = sum([contact_force[k](t)[i] for k in range(num_limbs)])
                self.AddLinearConstraint(total_contact_force + gravity_force[i] == robot.mass * acom(t)[i])

    def add_no_force_at_distance_constraints(self, contact, contact_force, Mbig):
        ts = contact.breaks
        dim = contact_force(ts[0]).size
        for t in ts[:-1]:
            if contact(t).dtype == np.object:
                for i in range(dim):
                    self.AddLinearConstraint(contact_force(t)[i] <= Mbig * contact(t)[0])
                    self.AddLinearConstraint(-contact_force(t)[i] <= Mbig * contact(t)[0])
            else:
                c = bool(round(contact(t)[0]))
                if not c:
                    for i in range(dim):
                        self.AddLinearConstraint(contact_force(t)[i] == 0)

    def add_contact_surface_constraints(self, qlimb, surface, contact, Mbig):
        """
        Add constraint that if contact(t) = 1, i.e. in contact during jth time step
        then qlimb must be on surface at t and t + dt
        :param qlimb:
        :param surface:
        :param contact:
        :param Mbig:
        :return:
        """
        ts = qlimb.breaks
        for j in range(len(ts) - 1):
            t = ts[j]
            A = surface.pose_constraints.getA()
            b = surface.pose_constraints.getB()
            qlimb_after_dt = qlimb.from_below(ts[j + 1])
            qlimb_current = qlimb(t)

            for i in range(A.shape[0]):
                self.AddLinearConstraint(A[i, :].dot(qlimb_current) <= b[i] + Mbig * (1 - contact(t)[0]))
                self.AddLinearConstraint(A[i, :].dot(qlimb_after_dt) <= b[i] + Mbig * (1 - contact(t)[0]))


            # if contact(t).dtype == np.object:
            #     for i in range(A.shape[0]):
            #         self.AddLinearConstraint(A[i, :].dot(qlimb_after_dt) <= b[i] + Mbig * (1 - contact(t)[0]))
            # else:
            #     c = bool(round(contact(t)[0]))
            #     if c:
            #         for i in range(A.shape[0]):
            #             self.AddLinearConstraint(A[i, :].dot(qlimb_after_dt) <= b[i])

    def add_contact_force_constraints(self, contact_force, surface, contact, Mbig):
        ts = contact_force.breaks
        for t in ts[:-1]:
            A = surface.force_constraints.getA()
            b = surface.force_constraints.getB()

            if contact(t).dtype == np.object:
                for i in range(A.shape[0]):
                    self.AddLinearConstraint(A[i, :].dot(contact_force(t)) <= b[i] + Mbig * (1 - contact(t)[0]))
            else:
                c = bool(round(contact(t)[0]))
                if c:
                    for i in range(A.shape[0]):
                        self.AddLinearConstraint(A[i, :].dot(contact_force(t)) <= b[i])

    def add_contact_velocity_constraints(self, qlimb, contact, Mbig):
        ts = qlimb.breaks
        dim = qlimb(0).size

        vlimb = qlimb.derivative()
        for j in range(len(ts) - 2):
            t = ts[j]
            tnext = ts[j + 1]
            if contact(t).dtype == np.object:
                indicator = contact(t)[0]
                for i in range(dim):
                    self.AddLinearConstraint(vlimb(tnext)[i] <= Mbig * (1 - indicator))
                    self.AddLinearConstraint(-vlimb(tnext)[i] <= Mbig * (1 - indicator))
            else:
                indicator = bool(round(contact(t)[0]))
                if indicator:
                    for i in range(dim):
                        self.AddLinearConstraint(vlimb(tnext)[i] == 0)

    def get_piecewise_solution(self, piecewise):
        def get_solution(x):
            # this try/except block handles the case when x might be assigned to specific
            # value rather than being an optimization variable?
            try:
                return self.GetSolution(x)
            except TypeError:
                return x

        return piecewise.map(lambda p: p.map(lambda x: get_solution(x)))

    def extract_solution(self, robot, qcom, qlimb, contact, contact_force, contact_lambda,
                         contact_sequence_array):

        qcom = self.get_piecewise_solution(qcom)
        vcom = qcom.map(Polynomial.derivative)
        qlimb = [self.get_piecewise_solution(q) for q in qlimb]
        vlimb = [q.map(Polynomial.derivative) for q in qlimb]
        flimb = [self.get_piecewise_solution(f) for f in contact_force]
        contact =self.extract_contact_solution(contact, contact_lambda, contact_sequence_array)


        # note that flimb, vlimb are lists so we have to be extra careful when constructing
        # the Trajectory object for BoxAtlasInput
        return (Trajectory(
            [qcom, vcom] + qlimb,
            lambda qcom, vcom, *qlimb: BoxAtlasState(robot, qcom=qcom, vcom=vcom, qlimb=qlimb)
        ), Trajectory(
            [flimb, vlimb],
            lambda flimb, vlimb: BoxAtlasInput(robot, flimb=flimb, vlimb=vlimb)
        ), contact)

    def extract_contact_solution(self, contact, contact_lambda, contact_sequence_array):
        contact_val = [None]*len(contact)

        for idx, c in enumerate(contact):
            if contact_lambda[idx] is None:
                contact_val[idx] = self.get_piecewise_solution(c)
            else:
                # need special logic due to lambda formulation of contact
                contact_lambda_val = self.GetSolution(contact_lambda[idx])
                # print("contact_lambda_val", type(contact_lambda_val))
                # print("contact_lambda_val.dtype, ", contact_lambda_val.dtype)
                # print("np.shape(contact_lambda_val), ", np.shape(contact_lambda_val))
                ts = c.breaks
                contact_val[idx] = LambdaContactFormulation.constructContactPiecewisePolynomial(ts, contact_lambda_val, contact_sequence_array[idx])


        return contact_val


    def add_kinematic_constraints(self, qlimb, qcom, polytope):
        A = polytope.getA()
        b = polytope.getB()
        for (ql, qc) in islice(zip(qlimb.at_all_breaks(), qcom.at_all_breaks()), 1, None):
            offset = ql - qc
            for i in range(A.shape[0]):
                self.AddLinearConstraint((A[i, :].dot(offset) - b[i]) <= 0)

    def count_contact_switches(self, contact):
        ts = contact.breaks
        delta = self.NewContinuousVariables(len(ts) - 2, "contact_delta")
        for j in range(len(ts) - 2):
            self.AddLinearConstraint(contact(ts[j + 1])[0] - contact(ts[j])[0] <= delta[j])
            self.AddLinearConstraint(-(contact(ts[j + 1])[0] - contact(ts[j])[0]) <= delta[j])
        total_switches = self.NewContinuousVariables(1, "contact_switches")[0]
        self.AddLinearConstraint(sum(delta) <= total_switches)
        return total_switches


class BoxAtlasVariables(object):
    def __init__(self, prog, ts, num_limbs, dim, initial_state, contact_assignments=None, options=None):
        """

        :param prog:
        :param ts:
        :param num_limbs:
        :param dim:
        :param initial_state:
        :param contact_assignments: Dict with key = limb_idx, val = contact_assigment
        for that set of binary variables. contact_assigment must be a list with
        length = num_time_steps = len(ts) - 1. If value of
        val = contact_assigment[idx] is not None, then the constraint that the
        appropriate binary variable be equal to 'val' is added. Note that this will
        result in that binary variable being removed from the optimization during
        the presolve.
        :param options:
        """

        if options is None:
            options = dict()
            options['use_lambda_contact_fomulation'] = False


        self.qcom = prog.new_piecewise_polynomial_variable(ts, dim, 2)
        prog.add_continuity_constraints(self.qcom)
        self.vcom = self.qcom.derivative()
        prog.add_continuity_constraints(self.vcom)

        self.qlimb = [prog.new_piecewise_polynomial_variable(ts, dim, 1) for k in range(num_limbs)]
        self.vlimb = [q.derivative() for q in self.qlimb]
        for q in self.qlimb:
            prog.add_continuity_constraints(q)
        self.contact_force = [prog.new_piecewise_polynomial_variable(ts, dim, 0) for k in range(num_limbs)]
        self.contact= [prog.new_piecewise_polynomial_variable(ts, 1, 0, kind="binary") for k in range(num_limbs)]


        # if contact_assignments was passed in then constrain those variables
        if contact_assignments is not None:
            print("contact assigments passed in")
            num_time_steps = len(ts) - 1
            for limb_idx, contact_sequence in contact_assignments.iteritems():
                contact_vars = self.contact[limb_idx]
                assert(len(contact_sequence) == num_time_steps)
                for idx, c in enumerate(contact_vars.at_all_breaks()):
                    # don't index past end of contact_sequence list
                    c = c[0]
                    if (idx > num_time_steps - 1):
                        break

                    # if the value of the assignment is not None, then
                    # add a constraint to the MIQP that pins down that
                    # binary variable
                    val = contact_sequence[idx]
                    if val is not None:
                        assert(val in [0,1])
                        prog.AddLinearConstraint(c == val)



        # old stuff
        self.contact_lambda = [None] * num_limbs
        self.contact_sequence_array = [None]*num_limbs
        # DEPRECATED

        # if contact_assignments is None:
        #     contact_assignments = [None for i in range(num_limbs)]
        #
        # assert len(contact_assignments) == num_limbs
        # self.contact = [None]*num_limbs
        # self.contact_lambda = [None]*num_limbs
        # self.contact_sequence_array = [None]*num_limbs
        #
        # for idx, c in enumerate(contact_assignments):
        #     if c is None:
        #         if options['use_lambda_contact_formulation']:
        #             # this is lambda formulation where we enumerate potential contact sequences
        #             initial_contact_state = initial_state.contact_indicator[idx]
        #             contact_vars = LambdaContactFormulation.addContactVariables(prog,
        #                                                                         ts,
        #                                                                         initial_contact_state)
        #             self.contact_lambda[idx] = contact_vars.contact_lambda
        #             self.contact[idx] = contact_vars.contact
        #             self.contact_sequence_array[idx] = contact_vars.contact_sequence_array
        #         else: # standard formulation
        #             self.contact[idx] = prog.new_piecewise_polynomial_variable(ts, 1, 0, kind="binary")
        #     else:
        #         self.contact[idx] = c # c is a PiecewisePolynomial here

    def all_state_variables(self):
        x = []
        for j in range(len(self.qcom.functions)):
            xj = np.hstack([np.hstack(self.qcom.functions[j].coeffs[:-1])] +
                           [np.hstack(q.functions[j].coeffs[:-1]) for q in self.qlimb])
            x.append(xj)
        return np.vstack(x).T

    def all_input_variables(self):
        u = []
        for j in range(len(self.qcom.functions)):
            uj = np.hstack([np.hstack(self.qcom.functions[j].coeffs[-1:])] +
                           [np.hstack(q.functions[j].coeffs[-1:]) for q in self.qlimb] +
                           [np.hstack(f.functions[j].coeffs) for f in self.contact_force])
            u.append(uj)
        return np.vstack(u).T


class LambdaContactFormulation(object):
    @staticmethod
    def addContactVariables(prog, ts, initial_contact_state):
        """

        :param prog: MathematicalProgram inside BoxAtlasContactStabilization
        :param ts:
        :param initial_state:
        :param limb_idx:
        :param vars: BoxAtlasVarriables
        :return:
        """
        num_time_steps = len(ts) - 1
        contact_sequence_array = LambdaContactFormulation.enumerateContactSequences(num_time_steps, initial_contact_state)

        # vars will be called contact_lambda
        # one binary variable for each potential contact sequence
        num_contact_sequences = np.shape(contact_sequence_array)[0]
        contact_lambda = prog.NewBinaryVariables(1, num_contact_sequences)
        contact_lambda = np.reshape(contact_lambda, [np.size(contact_lambda)])

        # add constraint that they sum to one
        prog.AddLinearConstraint(np.sum(contact_lambda) == 1)

        # now must create contact variables from these binary vars + contact_sequence_array

        contact = LambdaContactFormulation.constructContactPiecewisePolynomial(ts, contact_lambda, contact_sequence_array)

        contact_vars = ContactVariables(contact_lambda=contact_lambda,
                                        contact=contact,
                                        contact_sequence_array=contact_sequence_array)
        return contact_vars

    @staticmethod
    def enumerateContactSequences(num_time_steps, initial_contact_state):
        """
        Enumerates all the potential "good" contact sequences, i.e. at most one
        contact switch
        :param num_time_steps:
        :param initial_contact_state: whether we are in contact or not at first timestep
        :return: List of np.array, each one is a potential contact sequence
        """

        def getOtherContactState(x):
            if x==0:
                return 1
            elif x==1:
                return 0
            else:
                raise ValueError("x must be 0 or 1")

        other_contact_state = getOtherContactState(initial_contact_state)

        # initialize contact sequence list
        contact_sequence_list = []

        # add sequence which just stays the as the initial_contact_state for all time
        # this one is a corner case
        contact_sequence_list.append(initial_contact_state* np.ones(num_time_steps, dtype=int))

        for i in xrange(1,num_time_steps):
            for j in xrange(1, num_time_steps + 1 - i):
                # print("(i,j) = (%s, %s)" % (i,j))
                cs = initial_contact_state*np.ones(num_time_steps, dtype=int)
                cs[i:i+j] = other_contact_state
                contact_sequence_list.append(cs)

        contact_sequence_array = np.array(contact_sequence_list)

        return contact_sequence_array

    @staticmethod
    def constructContactPiecewisePolynomial(ts, contact_lambda, contact_sequence_array):
        contact_piecewise_functions = []
        for i in xrange(len(ts) - 1):
            # value at knot point 'i' is sum of values from different lambas
            val = np.dot(contact_sequence_array[:, i], contact_lambda)
            # have to make the value into a polynomial so that it is callable
            polynomial_val = Polynomial(np.array([[val]]))
            contact_piecewise_functions.append(polynomial_val)

        contact = Piecewise(ts, contact_piecewise_functions)
        return contact


class BoxAtlasContactStabilization(object):
    def __init__(self, robot, initial_state, env, desired_state,
                 dt=0.05,
                 num_time_steps=20,
                 params=None,
                 contact_assignments=None, options=None):

        # load the parameters
        if params is None:
            self.params = BoxAtlasContactStabilization.get_optimization_parameters()
        else:
            self.params = params

        if options is None:
            self.options = BoxAtlasContactStabilization.make_default_options()
        else:
            self.options = options

        self.robot = robot
        self.num_time_steps = num_time_steps
        time_horizon = num_time_steps * dt
        self.dt = dt
        self.ts = np.linspace(0, time_horizon, num_time_steps + 1)
        self.dim = self.robot.dim

        self.env = env
        self.prog = MixedIntegerTrajectoryOptimization()
        num_limbs = len(self.robot.limb_bounds)
        self.vars = BoxAtlasVariables(self.prog, self.ts, num_limbs, self.dim, initial_state,
                                      contact_assignments, options=self.options)
        self.add_constraints()
        if initial_state is not None:
            self.add_initial_state_constraints(initial_state)
        self.add_costs(desired_state)

    def add_initial_state_constraints(self, initial_state):
        num_limbs = len(self.robot.limb_bounds)
        for i in range(self.dim):
            self.prog.AddLinearConstraint(self.vars.qcom(self.ts[0])[i] == initial_state.qcom[i])
            self.prog.AddLinearConstraint(self.vars.vcom(self.ts[0])[i] == initial_state.vcom[i])
            for k in range(num_limbs):
                # don't constrain initial limb velocity to be zero
                self.prog.AddLinearConstraint(self.vars.qlimb[k](self.ts[0])[i] == initial_state.qlimb[k][i])

                if self.params['options']['zero_initial_limb_velocity']:
                    self.prog.AddLinearConstraint(self.vars.vlimb[k](self.ts[0])[i] == 0)

    def add_contact_switch_constraints(self, max_num_switches=2):
        num_limbs = len(self.robot.limb_bounds)
        switches = [None]*num_limbs
        for k in xrange(num_limbs):
            # only add this constraint if those contact variables haven't been assigned
            if self.vars.contact[k](0).dtype == np.object:
                switches[k] = self.prog.count_contact_switches(self.vars.contact[k])
                self.prog.AddLinearConstraint(switches[k] <= max_num_switches)

    def add_one_foot_on_ground_constraint(self):
        """
        Enforces that at least one foot be on the ground at all times
        :return: None
        """
        left_leg_idx = self.robot.limb_idx_map["left_leg"]
        right_leg_idx = self.robot.limb_idx_map["right_leg"]

        for t in self.ts[:-1]:
            left_leg_contact = self.vars.contact[left_leg_idx](t)
            right_leg_contact = self.vars.contact[right_leg_idx](t)

            # only add this constraint if at least one of these is an object
            if (left_leg_contact.dtype == np.object) or (right_leg_contact.dtype == np.object):
                self.prog.AddLinearConstraint(left_leg_contact[0] + right_leg_contact[0] >= 1)


    def add_constraints(self, vlimb_max=5, Mq=10, Mv=100, Mf=1000):
        num_limbs = len(self.robot.limb_bounds)
        for k in range(num_limbs):

            self.prog.add_no_force_at_distance_constraints(self.vars.contact[k],
                                                           self.vars.contact_force[k], Mf)
            self.prog.add_contact_surface_constraints(self.vars.qlimb[k],
                                                      self.env.surfaces[k],
                                                      self.vars.contact[k], Mq)
            self.prog.add_contact_force_constraints(self.vars.contact_force[k],
                                                    self.env.surfaces[k],
                                                    self.vars.contact[k], Mf)
            self.prog.add_contact_velocity_constraints(self.vars.qlimb[k],
                                                       self.vars.contact[k], Mv)

            vlimb_max = self.robot.limb_velocity_limits[k]
            self.prog.add_limb_velocity_constraints(self.vars.qcom,
                                                    self.vars.qlimb[k],
                                                    vlimb_max,
                                                    self.dt)
            self.prog.add_kinematic_constraints(self.vars.qlimb[k],
                                                self.vars.qcom,
                                                self.robot.limb_bounds[k])
            # switches = self.prog.count_contact_switches(self.vars.contact[k])
            # self.prog.AddLinearConstraint(switches <= 2)
        # for k in [1, 2]:
        #     for t in self.vars.qlimb[k].breaks[:-1]:
        #         self.prog.AddLinearConstraint(self.vars.contact[k](t)[0] == 1)
        #         # self.prog.AddLinearConstraint(self.vars.qlimb[k](t)[1] >= 0.0 * (1 - self.vars.contact[k](t)[0]))

        for f in chain(*[fl.at_all_breaks() for fl in self.vars.contact_force]):
            for i in range(self.dim):
                self.prog.AddLinearConstraint(f[i] <= Mf)
                self.prog.AddLinearConstraint(f[i] >= -Mf)

        for q in chain(self.vars.qcom.at_all_breaks(),
                       *[ql.at_all_breaks() for ql in self.vars.qlimb]):
            for i in range(self.dim):
                self.prog.AddLinearConstraint(q[i] <= 10)
                self.prog.AddLinearConstraint(q[i] >= -10)

        for v in chain(self.vars.qcom.derivative().at_all_breaks(),
                       *[ql.derivative().at_all_breaks() for ql in self.vars.qlimb]):
            for i in range(self.dim):
                self.prog.AddLinearConstraint(v[i] <= Mv)
                self.prog.AddLinearConstraint(v[i] >= -Mv)

        for a in self.vars.qcom.derivative().derivative().at_all_breaks():
            for i in range(self.dim):
                self.prog.AddLinearConstraint(a[i] <= Mf)
                self.prog.AddLinearConstraint(a[i] >= -Mf)

        A = self.env.free_space.getA()
        b = self.env.free_space.getB()
        for q in chain(self.vars.qcom.at_all_breaks(),
                       *[ql.at_all_breaks() for ql in self.vars.qlimb]):
            for i in range(A.shape[0]):
                self.prog.AddLinearConstraint((A[i, :].dot(q) - b[i]) <= 0)

        self.prog.add_dynamics_constraints(self.robot, self.vars.qcom, self.vars.contact_force)

    def add_costs(self, desired_state):
        num_limbs = len(self.robot.limb_bounds)
        cost_weights = self.params['costs']
        self.prog.AddQuadraticCost(
            cost_weights['contact_force'] * np.sum(np.sum(np.power(self.vars.contact_force[k](t), 2)) for t in self.ts[:-1] for k in range(num_limbs)))

        # add the running costs
        self.add_running_costs(desired_state)
        self.add_final_costs(desired_state)
        self.add_limb_velocity_cost(self.params['costs']['limb_velocity'])

    def add_final_costs(self, desired_state):
        cost_weights = self.params['costs']
        num_limbs = len(self.robot.limb_bounds)
        qcomf = self.vars.qcom.from_below(self.ts[-1])
        vcomf = self.vars.vcom.from_below(self.ts[-1])
        self.prog.AddQuadraticCost(
            cost_weights['qcom_final'] * np.sum(np.power(qcomf - desired_state.qcom, 2)))

        self.prog.AddQuadraticCost(
            cost_weights['vcom_final'] * np.sum(np.power(vcomf - desired_state.vcom, 2)))


        # limb final position costs
        qlimbf = [self.vars.qlimb[k].from_below(self.ts[-1]) for k in range(num_limbs)]
        right_arm_idx = self.robot.limb_idx_map["right_arm"]
        right_leg_idx = self.robot.limb_idx_map["right_leg"]
        left_arm_idx = self.robot.limb_idx_map["left_arm"]
        left_leg_idx = self.robot.limb_idx_map["left_leg"]

        self.prog.AddQuadraticCost(
            cost_weights["arm_final_position"] * np.sum(np.power(qlimbf[right_arm_idx] - desired_state.qlimb[right_arm_idx], 2)))
        self.prog.AddQuadraticCost(
            cost_weights["arm_final_position"] * np.sum(np.power(qlimbf[left_arm_idx] - desired_state.qlimb[left_arm_idx], 2)))

        # final position costs for legs
        self.prog.AddQuadraticCost(
            cost_weights['leg_final_position'] * np.sum(np.power(qlimbf[right_leg_idx] - desired_state.qlimb[right_leg_idx], 2)))
        self.prog.AddQuadraticCost(
            cost_weights['leg_final_position'] * np.sum(np.power(qlimbf[left_leg_idx] - desired_state.qlimb[left_leg_idx], 2)))


    def add_running_costs(self, desired_state):
        cost_weights = self.params['costs']
        self.prog.AddQuadraticCost(
            cost_weights['qcom_running'] * np.sum(
                np.sum(np.power(q - desired_state.qcom, 2)) for q in self.vars.qcom.at_all_breaks()))

        self.prog.AddQuadraticCost(
            cost_weights['vcom_running'] * np.sum(
                [np.sum(np.power(q.derivative()(t), 2) for t in self.ts[:-1]) for q in self.vars.qlimb]))

        # add running costs for limb position deviation from nominal
        self.add_limb_running_costs(desired_state)

    def add_limb_running_costs(self, desired_state):
        weight = self.params['costs']['limb_running']
        qcom = self.vars.qcom
        dt = self.dt
        ts = qcom.breaks

        num_limbs = len(self.vars.qlimb)

        for limb_idx in xrange(0, num_limbs):
            limb_com_desired = desired_state.qlimb[limb_idx] - desired_state.qcom
            qlimb = self.vars.qlimb[limb_idx]
            for j in range(len(ts) - 1):
                limb_com = qlimb(ts[j]) - qcom(ts[j])

                self.prog.AddQuadraticCost(weight*np.sum(np.power(limb_com - limb_com_desired, 2)))


    def add_limb_velocity_cost(self, weight):
        qcom = self.vars.qcom
        dt = self.dt

        for qlimb in self.vars.qlimb:
            ts = qcom.breaks
            dim = qcom(ts[0]).size
            vcom = qcom.derivative()
            for j in range(len(ts) - 2):
                relative_velocity = 1.0 / dt * (qlimb(ts[j + 1]) - qlimb(ts[j])) - vcom(ts[j])
                for i in range(dim):
                    self.prog.AddQuadraticCost(weight*(relative_velocity[i])**2)
    def solve(self):
        solver = GurobiSolver()
        self.prog.SetSolverOption(mp.SolverType.kGurobi, "LogToConsole", 1)
        self.prog.SetSolverOption(mp.SolverType.kGurobi, "OutputFlag", 1)
        start_time = time.time()
        result = solver.Solve(self.prog)
        solve_time = time.time() - start_time
        assert result == mp.SolutionResult.kSolutionFound
        states, inputs, contact = self.prog.extract_solution(
            self.robot, self.vars.qcom, self.vars.qlimb,
            self.vars.contact, self.vars.contact_force, self.vars.contact_lambda,
        self.vars.contact_sequence_array)
        ts = states.components[0].breaks
        return SolutionData(opt=self, states=states, inputs=inputs,
                            contact_indicator=contact, ts=ts, solve_time=solve_time)

    @staticmethod
    def get_optimization_parameters():
        params = dict()

        # weights for all the costs in the optimization
        params['costs'] = dict()
        params['costs']['contact_force'] = 1e-2
        params['costs']['qcom_running'] = 1e3
        params['costs']['vcom_running'] = 1e3
        params['costs']['limb_running'] = 1
        params['costs']['qcom_final'] = 1e3
        params['costs']['vcom_final'] = 1e4
        params['costs']['arm_final_position'] = 1e4
        params['costs']['limb_velocity'] = 1e-1
        params['costs']['leg_final_position'] = 1e2

        params['options'] = dict()
        params['options']['zero_initial_limb_velocity'] = True
        return params

    @staticmethod
    def make_default_options():
        options = dict()
        options['use_lambda_contact_formulation'] = False
        return options

