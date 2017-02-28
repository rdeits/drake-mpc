@testset "construct optimization problem" begin
  param = OptimizationParameters(num_timesteps = 10)
  model = Model(solver=GurobiSolver(Presolve=0))
  p = CentroidalDynamicsOptimizationProblem(model=model, param=param)
end

@testset "contact_point" begin
  ContactPoint("l_foot",[-0.5,0,0])
end

@testset "add_contact_points!" begin
  p, weights, contact_points = construct_default_problem()
  add_contact_points!(p, contact_points)
end

@testset "add_variables!" begin
  p, weights, contact_points = construct_default_problem()
  add_variables!(p)
  add_contact_points!(p, contact_points)
end

@testset "add_dynamics_constraints!" begin
  p, weights, contact_points = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_dynamics_constraints!(p)
end

@testset "add_initial_condition_constraints!" begin
  p, weights, contact_points, initial_conditions = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_initial_condition_constraints!(p, initial_conditions)
end

@testset "add_costs!" begin
  p, weights, contact_points, initial_conditions = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_costs!(p, weights)
end

@testset "run_optimization" begin
  p, weights, contact_points, initial_conditions = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_dynamics_constraints!(p)
  add_initial_condition_constraints!(p, initial_conditions)
  add_costs!(p, weights)

  # solve the model
  status = solve(p.model)
end

@testset "get_variable_solution_values" begin
  p, weights, contact_points, initial_conditions = construct_default_problem()
  add_contact_points!(p, contact_points)
  add_variables!(p)
  add_dynamics_constraints!(p)
  add_initial_condition_constraints!(p, initial_conditions)
  add_costs!(p, weights)

  # solve the model
  status = solve(p.model)
  soln = get_variable_solution_values(p.vars)
end
