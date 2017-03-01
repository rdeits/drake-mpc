type ContactPoint
  name::String
  location::Array{Float64}
end

@with_kw type OptimizationParameters
  dimension::Int = 3
  num_timesteps::Int = 1
  robot_mass::Float64 = 1
  robot_inertia::Float64 = 1 # for debugging, not used in optimization
  dt::Float64 = 0.1
  gravity::Vector{Float64} = [0.,0.,-9.8] # change this if you change the dimension
end

@with_kw type OptimizationWeights
  com_final_position::Vector{Float64} = [0.,0.,1.]
  com_final_position_weight::Float64 = 100.
  lin_momentum::Float64 = 0.5
  forces::Float64 = 0.001
  convex_bounds::Float64 = 1e-3
end

@with_kw type OptimizationInitialConditions
  com_position::Vector{Float64} = [0.,0.,1.]
  lin_momentum::Vector{Float64} = zeros(3)
  ang_momentum::Vector{Float64} = zeros(3)
end

@with_kw type OptimizationVariables{T}
  # variables
  com_position::Array{T} = Array{T}()
  forces::Array{T} = Array{T}()
  torques::Array{T} = Array{T}()
  lin_momentum::Array{T} = Array{T}()
  lin_momentum_dot::Array{T} = Array{T}()
  ang_momentum::Array{T} = Array{T}()
  ang_momentum_dot::Array{T} = Array{T}()
  l_cross_f_plus::Array{T} = Array{T}()
  l_cross_f_minus::Array{T} = Array{T}()
end

@with_kw type OptimizationDebugValues
  pos_convex::Array{Float64} = Array{Float64}()
  neg_convex::Array{Float64} = Array{Float64}()
  l_cross_f::Array{Float64} = Array{Float64}() # torque contribution for each contact point
  ang_momentum::Array{Float64} = Array{Float64}()
  ang_momentum_dot::Array{Float64} = Array{Float64}()
  orientation::Array{Float64} = Array{Float64}()
end

# store the optimization problem
@with_kw type CentroidalDynamicsOptimizationProblem
  model::JuMP.Model
  param::OptimizationParameters = OptimizationParameters()
  contact_points::Vector{ContactPoint} = Vector{ContactPoint}()
  vars::OptimizationVariables{JuMP.Variable} = OptimizationVariables{JuMP.Variable}()
end

type OptimizationSolution
  com_position::Array{Float64}
  forces::Array{Float64}
  torques::Array{Float64}
  lin_momentum::Array{Float64}
  lin_momentum_dot::Array{Float64}
  ang_momentum::Array{Float64}
  ang_momentum_dot::Array{Float64}
  l_cross_f_plus::Array{Float64}
  l_cross_f_minus::Array{Float64}

  # parse the solution from a CentroidalDynamicsOptimizationProblem
  function OptimizationSolution(p::CentroidalDynamicsOptimizationProblem)
    var_list = [:com_position, :forces, :torques, :lin_momentum,
    :lin_momentum_dot, :ang_momentum_dot, :l_cross_f_plus, :l_cross_f_minus]
    soln = new() # uninitialized object

    # for each variable get the JuMP.Variable array from p.
    # get the value and then set it to the correct field
    for var_symbol in var_list
      var_value = getvalue(getfield(p,var_symbol))
      setfield!(soln, var_symbol, var_value)
    end
    return soln
  end
end

# returns the square of the L2 norm of a vector or matrix
function l2_norm(x)
  return sum(x.^2)
end

# helper function for getting difference of convex functions
# decomposition of cross product
function difference_convex_functions_decomposition(l, force)
  # l = p - r, ray from com to contact location

  # the commas are important here to have size (2,) rather
  # than (1,2)
  a = [-l[3],l[2]]
  b = [l[3],-l[1]]
  c = [-l[2],-l[1]]
  d = force[2:3]
  e = force[[1,3]]
  f = force[1:2]

  # this is the decomposition of the cross product into difference
  # of convex functions
  # these are 3-vectors
  p = 1/4 * [l2_norm(a.+d),l2_norm(b.+e),l2_norm(c.+f)]
  q = 1/4 * [l2_norm(a.-d),l2_norm(b.-e),l2_norm(c.-f)]

  return p,q
end

function add_contact_points!(p::CentroidalDynamicsOptimizationProblem, contact_points::Vector{ContactPoint})
  append!(p.contact_points, contact_points)
end

function add_variables!(p::CentroidalDynamicsOptimizationProblem)
  num_knot_points = p.param.num_timesteps+1
  num_contacts = length(p.contact_points)
  num_timesteps = p.param.num_timesteps
  dimension = p.param.dimension


  # com position
  p.vars.com_position = @variable(p.model, [1:dimension, 1:num_knot_points])

  # linear momentum
  p.vars.lin_momentum = @variable(p.model, [1:dimension, 1:num_knot_points])

  # angular momentum
  p.vars.ang_momentum = @variable(p.model, [1:dimension, 1:num_knot_points])

  # linear momentum dot
  p.vars.lin_momentum_dot = @variable(p.model, [1:dimension, 1:num_knot_points])

  # angular momentum dot
  p.vars.ang_momentum_dot = @variable(p.model, [1:dimension, 1:num_knot_points])

  # contact force variables
  p.vars.forces = @variable(p.model, [1:3, 1:num_timesteps, 1:num_contacts], lowerbound = 0)

  # contact torque variables
  # clamp them to zero for the moment
  p.vars.torques = @variable(p.model, [1:3, 1:num_timesteps, 1:num_contacts], lowerbound = 0,
  upperbound = 0)

  # bounding variables for l x f terms
  p.vars.l_cross_f_plus = @variable(p.model, [1:dimension, 1:num_timesteps, 1:num_contacts],
  lowerbound = 0)
  p.vars.l_cross_f_minus = @variable(p.model, [1:dimension, 1:num_timesteps, 1:num_contacts],
  lowerbound = 0)
end

function add_linear_momentum_dynamics_constraints!(p::CentroidalDynamicsOptimizationProblem)
  const num_timesteps = p.param.num_timesteps
  const dt = p.param.dt
  const robot_mass = p.param.robot_mass
  const gravity = p.param.gravity
  const num_contacts = length(p.contact_points)

  for i=1:num_timesteps
      # com position integration constraint
      @constraint(p.model, p.vars.com_position[:,i+1] .- (p.vars.com_position[:,i] .+
          dt/robot_mass.*p.vars.lin_momentum[:,i]) .== 0)

      # linear momentum integration constraint
      @constraint(p.model, p.vars.lin_momentum[:,i+1] .- (p.vars.lin_momentum[:,i]
              + p.vars.lin_momentum_dot[:,i]*dt) .== 0)

      total_force = zero(AffExpr) # total force
      for contact_idx =1:num_contacts
          total_force += p.vars.forces[:,i,contact_idx]
      end

      # F = ma
      @constraint(p.model, p.vars.lin_momentum_dot[:,i] - (robot_mass*gravity .+ total_force) .== 0)
    end
end

function add_angular_momentum_dynamics_constraints!(p::CentroidalDynamicsOptimizationProblem)

  const num_timesteps = p.param.num_timesteps
  const dt = p.param.dt
  const num_contacts = length(p.contact_points)

  for i=1:num_timesteps
      # angular momentum Euler integration constraint
      @constraint(p.model, p.vars.ang_momentum[:, i+1] .- (p.vars.ang_momentum[:,i]
              .+ p.vars.ang_momentum_dot[:,i] * dt) .== 0)

      total_torque = zero(AffExpr) # total torque
      for contact_idx =1:num_contacts
          total_torque += p.vars.torques[:,i,contact_idx]
      end


      # the total l x f term
      l_cross_f = zero(AffExpr)
      for contact_idx =1:num_contacts
          # get contact location for this contact point
          contact_location = p.contact_points[contact_idx].location

          # vector from com --> contact location
          l = contact_location .- p.vars.com_position[:,i]
          force_local = p.vars.forces[:,i,contact_idx]

          # decompose into difference of convex functions
          pos_convex, neg_convex = difference_convex_functions_decomposition(l, force_local)

          # add constraint for convex relaxation
          # this is a quadratic constraint since pos_convex/neg_convex are
          # quadratic in the decision variables
          @constraint(p.model, pos_convex .<= p.vars.l_cross_f_plus[:,i,contact_idx])
          @constraint(p.model, neg_convex .<= p.vars.l_cross_f_minus[:,i,contact_idx])

          # add the contribution of this contact to the overall l x f term.
          l_cross_f += p.vars.l_cross_f_plus[:,i,contact_idx] - p.vars.l_cross_f_minus[:,i,contact_idx]
      end

      # angular momentum dot constraint
      @constraint(p.model, p.vars.ang_momentum_dot[:,i] .- (l_cross_f .+ total_torque) .== 0)
  end
end

function add_dynamics_constraints!(p::CentroidalDynamicsOptimizationProblem)
  add_linear_momentum_dynamics_constraints!(p)
  add_angular_momentum_dynamics_constraints!(p)
end

function add_initial_condition_constraints!(p::CentroidalDynamicsOptimizationProblem, x0::OptimizationInitialConditions)
  # initial condition constraints
  @constraint(p.model, p.vars.com_position[:,1] .== x0.com_position)
  @constraint(p.model, p.vars.lin_momentum[:,1] .== x0.lin_momentum)
  @constraint(p.model, p.vars.ang_momentum[:,1] .== x0.ang_momentum)
end

function add_costs!(p::CentroidalDynamicsOptimizationProblem, weights::OptimizationWeights)
  total_cost = get_running_cost(p,weights) + get_final_cost(p,weights)
  @objective(p.model, :Min, total_cost)
end

function get_final_cost(p::CentroidalDynamicsOptimizationProblem, weights::OptimizationWeights)
  com_final_position_cost = weights.com_final_position_weight*
  l2_norm(p.vars.com_position[:,end] - weights.com_final_position)
  return com_final_position_cost
end

function get_running_cost(p::CentroidalDynamicsOptimizationProblem, weights::OptimizationWeights)
  const num_contacts = length(p.contact_points)

  lin_momentum_cost = weights.lin_momentum*l2_norm(p.vars.lin_momentum[:,:,:])
  force_cost = zero(QuadExpr)
  convex_bound_cost = zero(QuadExpr)
  for contact_idx=1:num_contacts
      force_cost += weights.forces*l2_norm(p.vars.forces[:,:,contact_idx])
      convex_bound_cost += weights.convex_bounds*l2_norm(p.vars.l_cross_f_plus[:,:,contact_idx])
      convex_bound_cost += weights.convex_bounds*l2_norm(p.vars.l_cross_f_minus[:,:,contact_idx])
  end

  total_cost = lin_momentum_cost + force_cost + convex_bound_cost
  return total_cost
end

function get_variable_solution_values(vars::OptimizationVariables)
  """
  Parses model solution, essentially calls getvalue on each variable in
  our optimization problem
  """
  soln = OptimizationVariables{Float64}()

  for var_symbol in fieldnames(vars)
    setfield!(soln, var_symbol, getvalue(getfield(vars, var_symbol)))
  end

  return soln
end

function difference_convex_functions_decomposition(p::CentroidalDynamicsOptimizationProblem, soln::OptimizationVariables{Float64})

  const num_contacts = length(p.contact_points)
  const num_timesteps = p.param.num_timesteps
  pos_convex = 0*similar(soln.l_cross_f_plus)
  neg_convex = 0*similar(soln.l_cross_f_plus)

  for i=1:num_timesteps
    for contact_idx=1:num_contacts
      contact_location = p.contact_points[contact_idx].location

      # vector from com --> contact location
      l = contact_location .- soln.com_position[:,i]
      force_local = soln.forces[:,i,contact_idx]

      # decompose into difference of convex functions
      pos_convex[:,i,contact_idx], neg_convex[:,i,contact_idx] = difference_convex_functions_decomposition(l, force_local)
    end
  end

  return pos_convex, neg_convex
end

function construct_default_problem()
  param = OptimizationParameters()
  model = Model(solver=GurobiSolver(Presolve=0))
  p = CentroidalDynamicsOptimizationProblem(model=model, param=param)
  weights = OptimizationWeights()

  contact_points = Vector{ContactPoint}()
  push!(contact_points, ContactPoint("l_foot",[-0.5,0,0]))
  push!(contact_points, ContactPoint("r_foot",[0.5,0,0]))

  initial_conditions = OptimizationInitialConditions()
  return p, weights, contact_points, initial_conditions
end


function compute_debug_info(p::CentroidalDynamicsOptimizationProblem, soln::OptimizationVariables{Float64})

  const num_contacts = length(p.contact_points)
  const num_timesteps = p.param.num_timesteps
  const dt = p.param.dt
  const inertia = p.param.robot_inertia

  pos_convex = zeros(soln.l_cross_f_plus)
  neg_convex = zeros(soln.l_cross_f_minus)
  l_cross_f = zeros(soln.l_cross_f_minus)
  ang_momentum = zeros(soln.ang_momentum)
  ang_momentum_dot = zeros(soln.ang_momentum_dot)
  orientation = zeros(soln.ang_momentum)

  # setup initial condition for angular momentum
  ang_momentum[:,1] = soln.ang_momentum[:,1]

  for i=1:num_timesteps
    total_torque = zeros(ang_momentum[:,i])
    total_l_cross_f = zeros(ang_momentum[:,i])

    for contact_idx = 1:num_contacts
      # get contact location for this contact point
      contact_location = p.contact_points[contact_idx].location

      # vector from com --> contact location
      l = contact_location .- soln.com_position[:,i]
      force_local = soln.forces[:,i,contact_idx]

      # decompose into difference of convex functions
      pos_convex[:,i,contact_idx], neg_convex[:,i,contact_idx] = difference_convex_functions_decomposition(l, force_local)

      l_cross_f_local = cross(l,force_local)
      l_cross_f[:,i,contact_idx] = l_cross_f_local

      total_l_cross_f += l_cross_f_local
      total_torque += soln.torques[:,i,contact_idx]
    end

    ang_momentum_dot[:,i] = total_l_cross_f + total_torque
    ang_momentum[:,i+1] = ang_momentum[:,i] + dt*ang_momentum_dot[:,i]
    orientation[:,i+1] = orientation[:,i] + ang_momentum[:,i]*dt/inertia
  end

  debug_vals = OptimizationDebugValues(pos_convex=pos_convex, neg_convex=neg_convex,
  l_cross_f=l_cross_f, ang_momentum=ang_momentum, ang_momentum_dot=ang_momentum_dot,
  l_cross_f=l_cross_f, orientation=orientation)
end
