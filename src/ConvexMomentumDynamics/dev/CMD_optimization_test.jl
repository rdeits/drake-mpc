push!(LOAD_PATH, "../src/")
using ConvexMomentumDynamics
using JuMP
using Gurobi
using DrakeVisualizer


println("starting CMD optimization test")
param = OptimizationParameters(num_timesteps = 10)
model = Model(solver=GurobiSolver(Presolve=0))
p = CentroidalDynamicsOptimizationProblem(model=model, param=param)
weights = OptimizationWeights(com_final_position = [0.25,0.,2.])
weights.com_final_position_weight = 100
weights.lin_momentum = 1.
initial_conditions = OptimizationInitialConditions()

# create vector of contact points
contact_points = Vector{ContactPoint}()
push!(contact_points, ContactPoint("l_foot",[-0.5,0,0]))
push!(contact_points, ContactPoint("r_foot",[0.5,0,0]))

add_contact_points!(p, contact_points)
add_variables!(p)
add_dynamics_constraints!(p)
add_initial_condition_constraints!(p, initial_conditions)
add_costs!(p, weights)

# solve the model
status = solve(p.model)
soln = get_variable_solution_values(p.vars)
pos_convex, neg_convex = difference_convex_functions_decomposition(p, soln)

println("slack in difference of convex functions")
display(pos_convex - soln.l_cross_f_plus)

println("num contact points = ", length(p.contact_points))

println("final com position")
display(soln.com_position[:,end])

knot_point = 1
state = get_centroidal_dynamics_state(p, soln, knot_point)

## Visualize CentroidalDynamicsState
println("visualizing centroidal dynamics state")
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();

cd_vis = CentroidalDynamicsVisualizer(p=p, soln=soln)
delete!(cd_vis.vis)
draw_centroidal_dynamics_state(cd_vis, knot_point)

print("playing back trajectory")
playback_trajectory(cd_vis)

println("finished CMD optimization test")
