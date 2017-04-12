module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation
using CDDLib: CDDLibrary


include("types.jl")
include("utils.jl")
include("visualize.jl")

export
  Surface,
  Environment,
  BoxRobot,
  LimbConfig,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d,
  draw_environment
end
