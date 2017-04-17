module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation
using CDDLib: CDDLibrary


include("types.jl")
include("utils.jl")
include("visualize.jl")
include("simulate.jl")

export
  Surface,
  Environment,
  LimbConfig,
  BoxRobot,
  CentroidalDynamicsState,
  LimbState,
  BoxRobotState,
  LimbInput,
  BoxRobotInput,
  BoxRobotVisualizerOptions,
  simulate,
  h_representation_from_bounds,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d,
  draw_environment,
  draw_box_robot_state
end
