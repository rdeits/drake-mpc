module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation
using CDDLib: CDDLibrary


include("types.jl")
include("utils.jl")
include("control.jl")
include("simulate.jl")
include("visualize.jl")


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
  SimpleBoxAtlasControllerData,
  simple_controller_from_damping_ratio

# utils
export
  h_representation_from_bounds,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d


end
