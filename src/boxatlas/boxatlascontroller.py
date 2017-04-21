# Wrapper for BoxAtlasContactStabilization to facilitate feedback control
import numpy as np
import matplotlib.pyplot as plt

# custom imports
from irispy import Polyhedron
from utils.polynomial import Polynomial
from utils.piecewise import Piecewise
import boxatlas as box
from contactstabilization import BoxAtlasContactStabilization
from contactstabilizationutils import ContactStabilizationUtils as CSU
from contactstabilizationutils import BoxAtlasDefaults

class BoxAtlasController:

    def __init__(self, **kwargs):
        # initialize defaults
        self.defaults = BoxAtlasDefaults(**kwargs)
        BoxAtlasDefaults.fill_with_defaults(self.defaults)

    def construct_contact_stabilization_optimization(self, initial_state, options=None, contact_assignments=None, **kwargs):
        """
        Constructs a contact stabilization problem starting from the given initial state
        Can specify any additional information you want using kwargs which can be any
        of the ones that can be passed to BoxAtlasDefaults
        :param initial_state:
        :param kwargs:
        :return:
        """
        d = BoxAtlasDefaults.copy_with_kwargs(self.defaults, initial_state=initial_state,
                                              **kwargs)

        opt = BoxAtlasContactStabilization(d.robot, d.initial_state, d.env,
                                           d.desired_state, dt=d.dt,
                                           num_time_steps=d.num_time_steps,
                                           params=d.params,
                                           contact_assignments=contact_assignments,
                                           options=options)

        return opt