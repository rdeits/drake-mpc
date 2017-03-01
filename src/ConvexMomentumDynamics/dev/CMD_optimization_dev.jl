push!(LOAD_PATH, "../src/")
import ConvexMomentumDynamics

using ConvexMomentumDynamics
using JuMP
using Gurobi
using DrakeVisualizer


println("starting CMD optimization test")
param = OptimizationParameters(num_timesteps = 10)
model = Model(solver=GurobiSolver(Presolve=0))
p = CentroidalDynamicsOptimizationProblem(model=model, param=param)
weights = OptimizationWeights(com_final_position = [0.,0.,1.])
weights.convex_bounds = 1e-3
weights.com_final_position_weight = 100
weights.lin_momentum = 1.0
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
debug = compute_debug_info(p, soln)
println("pos_convex = ", debug.pos_convex[:,1,1])
println("l_cross_f_plus = ", soln.l_cross_f_plus[:,1,1])
println("convex gap = ", debug.pos_convex[:,1,1] - soln.l_cross_f_plus[:,1,1])
println("final com position ", soln.com_position[:,end])

knot_point = 1
state = get_centroidal_dynamics_state(p, soln, knot_point)

## Visualize CentroidalDynamicsState
println("visualizing centroidal dynamics state")
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();

cd_vis = CentroidalDynamicsVisualizer(p=p, soln=soln, debug=debug)
delete!(cd_vis.vis)
draw_centroidal_dynamics_state(cd_vis, knot_point)

print("playing back trajectory")
playback_trajectory(cd_vis)



println("finished CMD optimization test")
