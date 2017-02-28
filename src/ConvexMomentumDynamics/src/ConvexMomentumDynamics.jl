module ConvexMomentumDynamics

using JuMP
using Gurobi
using AxisArrays
using DrakeVisualizer
using CoordinateTransformations
using Parameters
using Interact, Reactive
import GeometryTypes: HyperRectangle, Vec, HomogenousMesh, Point
import ColorTypes: RGBA


export CentroidalDynamicsState
export CentroidalDynamicsOptimizationProblem
export ContactPoint
export OptimizationParameters
export OptimizationWeights
export OptimizationInitialConditions
export OptimizationVariables
export CentroidalDynamicsVisualizer
export CentroidalDynamicsVisualizerOptions

export add_variables!
export add_contact_points!
export add_dynamics_constraints!
export add_costs!
export add_initial_condition_constraints!
export get_variable_solution_values
export get_centroidal_dynamics_state
export draw_centroidal_dynamics_state
export playback_trajectory

# developed by Twan
# a more convenient method of using JuMP Variables than JuMP Arrays
# this is not really ready yet
macro axis_variables(m, var, axes...)
  for arg in axes
    @assert arg.head == :kw "Axis arguments must be of the form `name=domain`"
  end
  names = [arg.args[1] for arg in axes]
  domains = [arg.args[2] for arg in axes]
  ranges = [:(1:length($domain)) for domain in domains]
  axis_args = [Expr(:call, Expr(:curly, :Axis, Expr(:quote, names[i])), domains[i]) for i in eachindex(axes)]
  quote
    vars = @variable $m $var[$(ranges...)]
    $(esc(var)) = $(Expr(:call, :AxisArray, :vars, axis_args...))
  end
end

include("optimization.jl")
include("visualizer.jl")

end
