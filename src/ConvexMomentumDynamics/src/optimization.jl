type ContactPoint
  name::String
  location::Array{Float64}
end

@with_kw type OptimizationParameters
  dimension::Int = 3
  num_timesteps::Int = 1
  robot_mass::Float64 = 1
  dt::Float64 = 0.1
  gravity::Vector{Float64} = [0.,0.,-9.8] # change this if you change the dimension
end

# @with_kw type OptimizationWeights
#
# end

# store the optimization problem
@with_kw type CentroidalDynamicsOptimizationProblem
  model::JuMP.Model
  param::OptimizationParameters = OptimizationParameters()
  contact_points::Vector{ContactPoint} = Vector{ContactPoint}()

  # variables
  com_position::Array{JuMP.Variable} = Array{JuMP.Variable}()
  forces::Array{JuMP.Variable} = Array{JuMP.Variable}()
  torques::Array{JuMP.Variable} = Array{JuMP.Variable}()
  lin_momentum::Array{JuMP.Variable} = Array{JuMP.Variable}()
  lin_momentum_dot::Array{JuMP.Variable} = Array{JuMP.Variable}()
  ang_momentum::Array{JuMP.Variable} = Array{JuMP.Variable}()
  ang_momentum_dot::Array{JuMP.Variable} = Array{JuMP.Variable}()
  l_cross_f_plus::Array{JuMP.Variable} = Array{JuMP.Variable}()
  l_cross_f_minus::Array{JuMP.Variable} = Array{JuMP.Variable}()
end

# returns the square of the L2 norm of a vector
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
  p = 1/4 * [l2_norm(a+d),l2_norm(b+e),l2_norm(c+f)]
  q = 1/4 * [l2_norm(a-d),l2_norm(b-e),l2_norm(c-f)]

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
  p.com_position = @variable(p.model, [1:dimension, 1:num_knot_points])

  # linear momentum
  p.lin_momentum = @variable(p.model, [1:dimension, 1:num_knot_points])

  # angular momentum
  p.ang_momentum = @variable(p.model, [1:dimension, 1:num_knot_points])

  # linear momentum dot
  p.lin_momentum_dot = @variable(p.model, [1:dimension, 1:num_knot_points])

  # angular momentum dot
  p.ang_momentum_dot = @variable(p.model, [1:dimension, 1:num_knot_points])

  # contact force variables
  p.forces = @variable(p.model, [1:3, 1:num_timesteps, 1:num_contacts], lowerbound = 0)

  # contact torque variables
  # clamp them to zero for the moment
  p.torques = @variable(p.model, [1:3, 1:num_timesteps, 1:num_contacts], lowerbound = 0,
  upperbound = 0)

  # bounding variables for l x f terms
  p.l_cross_f_plus = @variable(p.model, [1:dimension, 1:num_timesteps, 1:num_contacts],
  lowerbound = 0)
  p.l_cross_f_minus = @variable(p.model, [1:dimension, 1:num_timesteps, 1:num_contacts],
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
      @constraint(p.model, p.com_position[:,i+1] .- (p.com_position[:,i] .+
          dt/robot_mass.*p.lin_momentum[:,i]) .== 0)

      # linear momentum integration constraint
      @constraint(p.model, p.lin_momentum[:,i+1] .- (p.lin_momentum[:,i]
              + p.lin_momentum_dot[:,i]*dt) .== 0)

      total_force = zero(AffExpr) # total force
      for contact_idx =1:num_contacts
          total_force += p.forces[:,i,contact_idx]
      end

      # F = ma
      @constraint(p.model, p.lin_momentum_dot[:,i] - (robot_mass*gravity .+ total_force) .== 0)
    end
end

function add_angular_momentum_dynamics_constraints!(p::CentroidalDynamicsOptimizationProblem)

  const num_timesteps = p.param.num_timesteps
  const dt = p.param.dt
  const num_contacts = length(p.contact_points)

  for i=1:num_timesteps
      # angular momentum Euler integration constraint
      @constraint(p.model, p.ang_momentum[:, i+1] .- (p.ang_momentum[:,i]
              .+ p.ang_momentum_dot[:,i] * dt) .== 0)

      total_torque = zero(AffExpr) # total torque
      for contact_idx =1:num_contacts
          total_torque += p.torques[:,i,contact_idx]
      end


      # the total l x f term
      l_cross_f = zero(AffExpr)
      for contact_idx =1:num_contacts
          # get contact location for this contact point
          contact_location = p.contact_points[contact_idx].location

          # vector from com --> contact location
          l = contact_location .- p.com_position[:,i]
          force_local = p.forces[:,i,contact_idx]

          # decompose into difference of convex functions
          pos_convex,neg_convex = difference_convex_functions_decomposition(l, force_local)

          # add constraint for convex relaxation
          @constraint(p.model, pos_convex .<= p.l_cross_f_plus[:,i,contact_idx])
          @constraint(p.model, neg_convex .<= p.l_cross_f_minus[:,i,contact_idx])

          # add the contribution of this contact to the overall l x f term.
          l_cross_f += p.l_cross_f_plus[:,i,contact_idx] - p.l_cross_f_minus[:,i,contact_idx]
      end

      # angular momentum dot constraint
      @constraint(p.model, p.ang_momentum_dot[:,i] .- (l_cross_f .+ total_torque) .== 0)
  end
end

function add_dynamics_constraints!(p::CentroidalDynamicsOptimizationProblem)
  add_linear_momentum_dynamics_constraints!(p)
  add_angular_momentum_dynamics_constraints!(p)
end
