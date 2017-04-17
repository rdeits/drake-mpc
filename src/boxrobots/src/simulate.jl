# templated on type of controller
type BoxRobotSimulationData{T <: BoxRobotController}
  t::Float64
  state::BoxRobotState
  input::BoxRobotInput
  controller_data::BoxRobotControllerData{T}
end

type BoxRobotSimulationDataArray
  tBreaks::Vector{Float64}
  data::Vector{BoxRobotSimulationData}
end
function simulate(robot::BoxRobot, state::BoxRobotState, input::BoxRobotInput, dt::Float64)
  """
  Simulate the system for dt seconds.
  Arguments:
    - robot: robot object being simulated
    - state: current state of the robot
    - input: the control input that will be applied during this simulation step]
    - dt: timespan (in seconds) to simulate
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

  # add the force from gravity
  total_force += robot.mass * robot.gravity

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
    if contained_in_h_representation(limb_config.surface.position, pos_next)
      pos_next = compute_non_penetrating_position(A,b,limb_state.pos,pos_next)
      vel_next = zeros(pos_next) # set velocity to zero
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
      warn("Limb $name currently in contact, but limb_input.acceleration doesn't move it out of contact, keeping it at it's current position")
    end

    limb_state_next = Base.deepcopy(limb_state)
    return limb_state_next
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

  eps_array = -ones(b)
  for (i,val) in enumerate(idx)
    # skip if A[i,:] x_prev <= b
    if !val
      continue
    end
    eps = (b_prev[i] - b[i])/(b_prev[i] - b_next[i])
    eps_array[i] = eps
  end

  # find smallest epsilon satisfying condition
  eps_min = minimum(eps_array[eps_array .> 0])

  x = (1-eps_min)*x_prev + eps_min*x_next
  return x
end


function simulate(robot::BoxRobot, controller::BoxRobotController, state::BoxRobotState, t::Float64, dt::Float64)
  """
  Simulates robot + controller system for dt seconds
  Essentially just a convenience method for calling
  simulate(robot::BoxRobot, state::BoxRobotState, input::BoxRobotInput, dt::Float64)

  Returns:
    - next_state
    - BoxRobotSimulationData
  """

  control_input, controller_data = compute_control_input(robot, controller, state, t, dt)
  next_state = simulate(robot, state, control_input, dt)
  simulation_data = BoxRobotSimulationData(t, state, control_input, controller_data)
  return next_state, simulation_data
end

function simulate_tspan(robot::BoxRobot, controller::BoxRobotController, initial_state::BoxRobotState, dt::Float64, duration::Float64)
  """
  Simulates robot + controller system
  Returns:
    - BoxRobotSimulationDataArray: contains all the information about the trajectory
  """
  num_time_steps = Int(ceil(duration/dt)) + 1
  tBreaks = dt*range(0,num_time_steps)
  data = Vector{BoxRobotSimulationData}(num_time_steps)
  state = initial_state
  for i=1:num_time_steps
    t = tBreaks[i]
    state, data[i] = simulate(robot, controller, state, t, dt)
  end

  return BoxRobotSimulationDataArray(tBreaks, data)

end
