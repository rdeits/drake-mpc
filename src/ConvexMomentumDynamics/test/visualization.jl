@testset "playback_trajectory" begin
  p, weights, contact_points, initial_conditions = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_dynamics_constraints!(p)
  add_initial_condition_constraints!(p, initial_conditions)
  add_costs!(p, weights)

  # solve the model
  status = solve(p.model)
  soln = get_variable_solution_values(p.vars)
  debug = compute_debug_info(p, soln)

  #playback trajectory
  DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
  cd_vis = CentroidalDynamicsVisualizer(p=p, soln=soln, debug=debug)
  playback_trajectory(cd_vis)
end
