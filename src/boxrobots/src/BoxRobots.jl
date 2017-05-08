module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation
using CDDLib: CDDLibrary


include("types.jl")
include("utils.jl")
include("control.jl")
include("controllers/MIQPcontroller.jl")
include("controllers/simpleinnerloopcontroller.jl")
include("controllers/QPinnerloopcontroller.jl")
include("simulate.jl")
include("visualize.jl")
include("defaults.jl")


# core
export
  Surface,
  Environment,
  LimbConfig,
  BoxRobot,
  CentroidalDynamicsState,
  LimbState,
  BoxRobotState,
  LimbInput,
  LimbInputType,
  ConstantVelocityLimbInput,
  ConstantAccelerationLimbInput,
  BoxRobotInput,
  ContactState

# visualization
export
  BoxRobotVisualizerOptions,
  draw_environment,
  draw_box_robot_state,
  playback_trajectory,
  slider_playback

# simulation
export
  simulate
  simulate_tspan

# controller
export
  BoxRobotController,
  BoxRobotControllerData,
  SimpleBoxAtlasController,
  MIQPController,
  SimpleBoxAtlasControllerData,
  SimpleInnerLoopController,

  simple_controller_from_damping_ratio
  convert_box_atlas_input_from_python,
  compute_control_input,

# QPInnerLoopController
export
  ContactSwitch,
  ContactSwitchType,
  MakingContact,
  BreakingContact,
  QPInnerLoopController,
  extract_contact_assignment_from_plan,
  compute_contact_assignment,
  convert_contact_assignment_to_python,
  make_contact_plan,
  get_contact_state_from_plan,

# utils
export
  h_representation_from_bounds,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d,
  contact_state_from_robot_state

# defaults
export
  make_robot_and_environment,
  make_robot_state,
  make_robot_input

end
