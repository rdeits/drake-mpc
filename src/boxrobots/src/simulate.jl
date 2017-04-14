function simulate(robot::BoxRobot, state::BoxRobotState, input::BoxRobotInput, dt::Float64)
  """
  Simulate the system for dt seconds.
  Arguments:
  Returns:
    next_state
  """

  centroidal_dynamics_state_next = simulate_centroidal_dynamics(robot, state.centroidal_dynamics_state, input, dt)

  limb_states_next = typeof(state.limb_states)() # Dict{Symbol, LimbState{T}}
  for (limb_sym, limb_state) in state.limb_states
    limb_config = robot.limbs[limb_sym]
    limb_input = input.limb_inputs[limb_sym]
    limb_state_next = simulate_limb_dynamics(limb_config, limb_state, limb_input, dt)
    limb_states_next[limb_sym] = limb_state_next
  end

  println("typeof(limb_states_next) = ", typeof(limb_states_next))
  state_next = BoxRobotState(centroidal_dynamics_state_next, limb_states_next)
end

function simulate_centroidal_dynamics(robot::BoxRobot, centroidal_dynamics_state::CentroidalDynamicsState, input::BoxRobotInput, dt::Float64)
  """
  Simulate the centroidal dynamics (just COM pos/vel for now) for dt seconds
  Returns:
    centroidal_dynamics_state_next
  """
  dim = robot.dim
  total_force = zeros(centroidal_dynamics_state.pos)

  for v in input.limb_inputs
    if v[2].has_force
      total_force += v[2].force
    end
  end

  # Euler integration
  com_acceleration = total_force/robot.mass
  pos_next = centroidal_dynamics_state.pos + centroidal_dynamics_state.vel*dt
  vel_next = centroidal_dynamics_state.vel + com_acceleration*dt

  centroidal_dynamics_state_next = CentroidalDynamicsState(pos_next, vel_next)
  return centroidal_dynamics_state_next
end

function simulate_limb_dynamics(limb_config::LimbConfig, limb_state::LimbState, limb_input::LimbInput, dt::Float64)
  """
  Simulate dynamics for a single limb for dt seconds
  Returns:
    limb_state_next
  """

  # if the limb is exerting force then it can't move, by convention
  # You can only apply force if the limb is currently in contact
  if limb_input.has_force
    if !limb_state.in_contact
      error("limb_input.has_force = True but limb_state.in_contact = False")
    end

    vel_next = zeros(limb_state.vel)
    pos_next = copy(limb_state.pos)
    in_contact = true
    limb_state_next = LimbState(pos_next, vel_next, in_contact)
    return limb_state_next
  end

  # if we have gotten here then the limb is not exerting force,
  # and so it can be moving. There are two cases, depending whether or not we
  # are currently in contact or not

  # extract SimpleHRepresentation of surface corresponding to this limb
  A = limb_config.surface.position.A
  b = limb_config.surface.position.b

  if !limb_state.in_contact
    # The limb is currently in free space. First check to see if we can move it
    # without penetrating the contact region
    pos_next = limb_state.pos + limb_state.vel*dt
    b_next = A*pos_next
    idx = b_next .< b

    # check if penetrates region
    if any(idx)
      pos_next = compute_non_penetrating_position(A,b,limb_state.pos,pos_next)
      vel_next = 0 # set velocity to zero
      in_contact = true
    else
      vel_next = limb_state.vel + limb_input.acceleration*dt
      in_contact = false
    end

    limb_state_next = LimbState(pos_next, vel_next, in_contact)
    return limb_state_next
  else
    # in this case we are currently in contact, so the acceleration/velocity should
    # move us out of contact. Note that we always have zero velocity when in contact
    # so we have to do something a little smarter than Euler integration HyperSphere

    # we "should" have limb_state.vel == 0 since we are in contact
    vel_next = limb_state.vel + limb_input.acceleration*dt
    pos_next = limb_state.pos + limb_state.vel*dt + 1/2.0*dt^2*limb_input.acceleration

    # now we must check that pos_next is actually in the free space
    b_next = A*pos_next # at least one entry of b_next[i] > b[i]
    if !any(b_next .> b)
      name = limb_config.name
      warning("Limb $name currently in contact, but limb_input.acceleration doesn't move it out of contact")
    end
  end
end


function compute_non_penetrating_position(A,b,x_prev,x_next)
  """
  The assumption is that A x doesn't satisfy Ax_prev <= b (i.e. we aren't on contact surface)
  But that A x_next <= b. Want to find x =  x_prev + eps*(x_next - x_prev) with smallest eps that
  satisfies A x <= b, i.e. x should be on the contact surface
  """

  # first check if A x_prev satisfies A x_prev <= b then just return that.
  # this shouldn't really happen and we should print a warning

  b_prev = A*x_prev
  b_next = A*x_next
  idx = b_prev .> b

  if !any(idx) # at least one of A[i,:] x _prev > b[i] should be satisfied
    warning("A x_prev <= b is already satisfied, returning x = x_prev")
    return x_prev
  end

  eps_array = -ones(idx)
  for (i,val) in enumerate(idx)
    # skip if A[i,:] x_prev <= b
    if !val
      continue
    end
    eps = (b_prev[i] - b[i])/(b_prev[i] - b_next[i])
    eps_array[i] = eps
  end

  # find smallest epsilon satisfying condition
  eps_min = min(eps_array[eps_array > 0])

  x = (1-eps_min)*x_prev + eps_min*x_next
  return x
end
