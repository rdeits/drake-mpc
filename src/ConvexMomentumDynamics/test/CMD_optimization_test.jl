push!(LOAD_PATH, "../src/")
import ConvexMomentumDynamics
const CMD = ConvexMomentumDynamics

using ConvexMomentumDynamics
using JuMP

println("starting CMD optimization test")
param = OptimizationParameters(num_timesteps = 10)
p = CentroidalDynamicsOptimizationProblem(model=Model(), param=param)

contact_points = Vector{ContactPoint}()
push!(contact_points, ContactPoint("l_foot",[-0.5,0,0]))
push!(contact_points, ContactPoint("r_foot",[0.5,0,0]))


add_contact_points!(p, contact_points)
add_variables!(p)
println("num contacts = ", length(p.contact_points))
add_dynamics_constraints!(p)

println("num contact points = ", length(p.contact_points))
println("size(com_position) ", size(p.com_position))

println("finished CMD optimization test")
