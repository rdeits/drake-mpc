# Wrapper for BoxAtlasContactStabilization to facilitate feedback control

# custom imports
from contactstabilization import BoxAtlasContactStabilization
from contactstabilizationutils import BoxAtlasDefaults


class BoxAtlasController:

    def __init__(self, **kwargs):
        # initialize defaults
        self.defaults = BoxAtlasDefaults.make_defaults(**kwargs)
        self.robot = self.defaults["robot"]

    def construct_contact_stabilization_optimization(self, initial_state, **kwargs):
        """
        Constructs a contact stabilization problem starting from the given initial state
        Can specify any additional information you want using kwargs which can be any
        of the ones that can be passed to BoxAtlasDefaults
        :param initial_state:
        :param kwargs:
        :return:
        """
        d = BoxAtlasDefaults.copy_with_kwargs(self.defaults,
                                              initial_state=initial_state,
                                              **kwargs)

        opt = BoxAtlasContactStabilization(d["robot"], d["initial_state"], d["env"],
                                           d["desired_state"], dt=d["dt"],
                                           num_time_steps=d["num_time_steps"],
                                           params=d["params"],
                                           contact_assignments=d["contact_assignments"],
                                           options=d["options"])

        return opt

    def compute_control_input(self, initial_state, **kwargs):
        opt = self.construct_contact_stabilization_optimization(initial_state, **kwargs)
        solnData = opt.solve()
        box_atlas_input = self.extract_control_input_from_soln(solnData)
        return box_atlas_input, solnData


