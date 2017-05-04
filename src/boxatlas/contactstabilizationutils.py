# contains useful helper functions for constructing ContactStabilization problems

# standard lib imports
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
# custom imports
from irispy import Polyhedron
from utils.polynomial import Polynomial
from utils.piecewise import Piecewise
import boxatlas as box
from contactstabilization import BoxAtlasContactStabilization


class BoxAtlasDefaults(object):

    @staticmethod
    def make_defaults(**kwargs):
        CSU = ContactStabilizationUtils
        defaults = dict()
        defaults["num_time_steps"] = 20
        defaults["dt"] = 0.05
        defaults["options"] = None
        defaults["contact_assignments"] = None
        defaults["robot"] = CSU.make_box_atlas()
        defaults["env"] = CSU.make_environment()
        defaults["initial_state"] = CSU.make_default_initial_state(defaults["robot"])
        defaults["desired_state"] = CSU.make_default_desired_state(defaults["robot"])
        defaults["params"] = CSU.get_default_optimization_parameters()

        defaults.update(kwargs)
        return defaults

    @staticmethod
    def copy_with_kwargs(d, **kwargs):
        """
        Copies d and overwrites anything in kwargs
        :param d:
        :param kwargs:
        :return:
        """
        d_copy = d.copy()
        d_copy.update(kwargs)
        return d_copy


def get_limb_idx_map():
    """
    dict mapping limb_name --> limb_idx
    :return:
    """
    limb_idx_map = dict()
    limb_idx_map["right_arm"] = 0
    limb_idx_map["right_leg"] = 1
    limb_idx_map["left_leg"] = 2
    limb_idx_map["left_arm"] = 3
    return limb_idx_map


class ContactStabilizationUtils:
    limb_idx_map = get_limb_idx_map()

    @staticmethod
    def make_box_atlas(relax_leg_limb_bounds=True):
        atlas = box.BoxAtlas()

        if relax_leg_limb_bounds:
            # relax kinematic constraints on the legs
            large_leg_limb_bound = Polyhedron.fromBounds([-2.0, -1.0], [2.0, -0.5])
            leg_names = ["left_leg", "right_leg"]
            for limb in leg_names:
                idx = ContactStabilizationUtils.limb_idx_map[limb]
                atlas.limb_bounds[idx] = large_leg_limb_bound

        return atlas

    @staticmethod
    def make_environment(dist_to_wall=None, mu_wall=0.25, mu_floor = 0.5):
        """
        Constructs a box atlas environment with the given parameters
        :param dist_to_wall: (optional) dict with keys ['left','right']
        :param mu_wall: friction coefficient for wall
        :param mu_floor: friction coefficient for floor
        :return: boxatlas.boxatlas.Environment
        """

        if dist_to_wall is None:
            dist_to_wall = dict()
            dist_to_wall['left'] = -0.5
            dist_to_wall['right'] = 0.5

        # construct the different wall surfaces, note that one needs to be careful with the
        # friction cones, they are all in world frame at the moment, should probably fix this
        right_wall_surface = box.Surface(Polyhedron.fromBounds([dist_to_wall['right'], 0], [dist_to_wall['right'], 2]),
                    Polyhedron(np.array([[mu_wall, -1], [mu_wall, 1]]), np.array([0, 0])))

        left_wall_surface = box.Surface(Polyhedron.fromBounds([dist_to_wall['left'], 0], [dist_to_wall['left'], 2]),
                    Polyhedron(np.array([[-mu_wall, -1], [-mu_wall, 1]]), np.array([0, 0])))

        floor_surface = box.Surface(Polyhedron.fromBounds([dist_to_wall["left"], 0], [dist_to_wall["right"], 0]),
                    Polyhedron(np.array([[-1, -mu_floor], [1, -mu_floor]]), np.array([0, 0])))

        surfaces = [None]*4
        surfaces[0] = right_wall_surface
        surfaces[1] = floor_surface
        surfaces[2] = floor_surface
        surfaces[3] = left_wall_surface

        env = box.Environment(surfaces, Polyhedron.fromBounds([dist_to_wall['left'], 0], [dist_to_wall['right'], 2]))
        return env

    @staticmethod
    def get_default_optimization_parameters():
        """
        Construct set of default optimization parameters that seem to work well
        :return:
        """
        params = BoxAtlasContactStabilization.get_optimization_parameters()
        params['costs']['contact_force'] = 1e-3
        params['costs']['qcom_running'] = 1
        params['costs']['vcom_running'] = 1
        params['costs']['limb_running'] = 1e1
        params['costs']['qcom_final'] = 1e3
        params['costs']['vcom_final'] = 1e4
        params['costs']['arm_final_position'] = 0
        params['costs']['limb_velocity'] = 1e-1
        params['costs']['leg_final_position'] = 1e2

        params['options'] = dict()
        params['options']['zero_initial_limb_velocity'] = False
        return params


    # DEPRECATED
    @staticmethod
    def make_contact_assignment_old(dt, num_time_steps, constrained_limbs=None):
        """
        @inputs
        constrained_limbs: should be a dict with limb_name and value to which it is constrained

        default:
            both feet to always be in contact,
            right hand never to be in contact.
        """

        if constrained_limbs is None:
            constrained_limbs = dict()
            constrained_limbs["right_leg"] = 1  # persistent contact
            constrained_limbs["left_leg"] = 1  # persistent contact
            constrained_limbs["right_arm"] = 0  # not in contact

        time_horizon = num_time_steps * dt
        ts = np.linspace(0, time_horizon, time_horizon / dt + 1)
        domain = ts

        limb_idx_map = ContactStabilizationUtils.limb_idx_map
        contact_assignments = [None] * len(limb_idx_map)

        for limb_name, val in constrained_limbs.iteritems():
            limb_idx = limb_idx_map[limb_name]
            contact_assignments[limb_idx] = Piecewise(domain,
                                                      [Polynomial(np.array([[val]]))
                                                       for j in range(len(domain) - 1)])
        return contact_assignments

    @staticmethod
    def make_contact_assignment(dt, num_time_steps, constrained_limbs=None):
        """
        @inputs
        constrained_limbs: should be a dict with limb_name and value to which it is constrained

        default:
            both feet to always be in contact,
            right hand never to be in contact.
        """

        if constrained_limbs is None:
            constrained_limbs = dict()
            constrained_limbs["right_leg"] = 1  # persistent contact
            constrained_limbs["left_leg"] = 1  # persistent contact
            constrained_limbs["right_arm"] = 0  # not in contact

        contact_assignments = dict()
        limb_idx_map = ContactStabilizationUtils.limb_idx_map

        for limb_name, val in constrained_limbs.iteritems():
            limb_idx = limb_idx_map[limb_name]
            contact_assignments[limb_idx] = [val]*num_time_steps

        return contact_assignments

    @staticmethod
    def make_default_initial_state(robot):
        """
        Makes a default initial state, has zero com velocity for now
        """
        initial_state = box.BoxAtlasState(robot)
        initial_state = box.BoxAtlasState(robot)
        initial_state.qcom = np.array([0, 1])
        initial_state.vcom = np.array([0, 0.])
        initial_state.qlimb = map(np.array, [[0.35, 1], [0.25, 0], [-0.25, 0], [-0.35, 1]])
        return initial_state

    @staticmethod
    def make_default_desired_state(robot):
        """
        Same as default initial state, but with zero initial velocity
        """
        CSU = ContactStabilizationUtils
        desired_state = CSU.make_default_initial_state(robot)
        desired_state.vcom = np.zeros(2)
        return desired_state

    @staticmethod
    def make_default_optimization_problem(robot, env=None, initial_state=None,
                                          desired_state=None, params=None):
        CSU = ContactStabilizationUtils
        if env is None:
            env = CSU.make_environment();

        if initial_state is None:
            initial_state = CSU.make_default_initial_state(robot)

        if desired_state is None:
            desired_state = CSU.make_default_desired_state(robot)

        if params is None:
            params = CSU.get_default_optimization_parameters()

        opt = BoxAtlasContactStabilization(robot, initial_state, env,
                                           desired_state, params=params)
        return opt

    @staticmethod
    def get_contact_indicator_variable(solnData, contact_name="left_arm"):
        CSU = ContactStabilizationUtils
        idx = CSU.limb_idx_map[contact_name]
        ts = solnData.ts
        return [solnData.contact_indicator[idx](t) for t in ts[:-1]]

    @staticmethod
    def plot_contact_indicator(solnData, contact_name="left_arm"):
        CSU = ContactStabilizationUtils
        contact_indicator_left_arm = CSU.get_contact_indicator_variable(solnData, contact_name=contact_name)
        ts = solnData.ts
        vcom_x = solnData.states(0).vcom[0]
        label = 'initial com vel = ' + str(vcom_x)
        plt.plot(ts[:-1], contact_indicator_left_arm, label=label, alpha=1.0)
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('contact indicator')
        plt.show()

    @staticmethod
    def make_box_atlas_defaults():
        CSU = ContactStabilizationUtils
        defaults = BoxAtlasDefaults()
        defaults.robot = CSU.make_box_atlas()
        defaults.env = CSU.make_environment()
        defaults.initial_state = CSU.make_default_initial_state(defaults.robot)
        defaults.desired_state = CSU.make_default_desired_state(defaults.robot)
        defaults.params = CSU.get_default_optimization_parameters()
        return defaults

    @staticmethod
    def populate_box_atlas_defaults(defaults, robot=None, env=None, initial_state=None,
                                    desired_state=None, params=None):

        new_defaults = BoxAtlasDefaults()
        # initialize defaults
        if robot is None:
            new_defaults.robot = defaults.robot
        else:
            new_defaults.robot = robot

        if env is None:
            new_defaults.env = defaults.env
        else:
            new_defaults.env = env

        if initial_state is None:
            new_defaults.initial_state = defaults.initial_state
        else:
            new_defaults.initial_state = initial_state

        if desired_state is None:
            new_defaults.desired_state = defaults.desired_state
        else:
            new_defaults.desired_state = desired_state

        if params is None:
            new_defaults.params = defaults.params
        else:
            new_defaults.params = params
