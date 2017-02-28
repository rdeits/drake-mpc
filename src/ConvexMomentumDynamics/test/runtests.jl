push!(LOAD_PATH, "../src/")
using ConvexMomentumDynamics
using JuMP
using Gurobi
using DrakeVisualizer
using Base.Test

include("optimization.jl")
include("visualization.jl")
