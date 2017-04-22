path_to_my_module = string(Base.source_dir(), "/../src/")
push!(LOAD_PATH, path_to_my_module)
reload("BoxRobots")
import BoxRobots
br = BoxRobots
using Base.Test
using Polyhedra: SimpleHRepresentation
using DrakeVisualizer

include("types.jl")
include("visualize.jl")
include("simulate.jl")
