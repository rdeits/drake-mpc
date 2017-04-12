module BoxRobots

using Parameters
using Polyhedra: SimpleHRepresentation


include("types.jl")
include("utils.jl")
include("visualize.jl")

export
  BoxRobot,
  LimbConfig,
  polyhedron_from_bounds,
  convert_polyhedron_to_3d
end
