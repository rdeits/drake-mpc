module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation
using CDDLib: CDDLibrary


include("types.jl")
include("utils.jl")
include("control.jl")
include("controllers/MIQPcontroller.jl")
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
  BoxRobotInput

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
  simple_controller_from_damping_ratio
  convert_box_atlas_input_from_python,
  compute_control_input

# utils
export
  h_representation_from_bounds,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d

# defaults
export
  make_robot_and_environment,
  make_robot_state,
  make_robot_input

end
