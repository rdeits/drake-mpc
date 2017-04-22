abstract BoxRobotController
abstract BoxRobotControllerData{T<:BoxRobotController}

# maps julia limb symbols to python indices, note that the indices are
# really the python indices + 1, since julia is 1-indexed and python is
# 0-indexed
global julia_to_python_limb_map = Dict(:right_hand =>1, :right_foot => 2,
:left_foot => 3, :left_hand => 4)

function convert_box_atlas_input_from_python(box_atlas_input_python)
  """
  Converts BoxAtlasInput from boxatlas.py to Julia type
  BoxRobotInput{Float64, ConstantVelocityLimbInput}
  """
  limb_inputs = Dict{Symbol, LimbInput{Float64, ConstantVelocityLimbInput}}()
  for (limb_sym, limb_python_idx) in julia_to_python_limb_map
    vel = box_atlas_input_python[:vlimb][limb_python_idx]
    force = box_atlas_input_python[:flimb][limb_python_idx]
    # need to be careful here, seems that
    # typeof(box_atlas_input_python[:force_indicator][limb_python_idx]) = Array{Bool, 0}
    # we need it to be a Bool
    has_force = any(box_atlas_input_python[:force_indicator][limb_python_idx])
    limb_inputs[limb_sym] = LimbInput(vel, force, has_force, ConstantVelocityLimbInput)
  end

  box_robot_input = BoxRobotInput(limb_inputs)
  return box_robot_input
end


type SimpleBoxAtlasController <: BoxRobotController
  com_pos_desired::Vector{Float64}
  K_p::Float64
  K_d::Float64
end

type SimpleBoxAtlasControllerData{T} <: BoxRobotControllerData{SimpleBoxAtlasController}
  t::T
  com_acc::Vector{T}
end

function simple_controller_from_damping_ratio(com_pos_desired, K_p, damping_ratio)
  K_d = 2.0*sqrt(K_p)*damping_ratio
  simple_box_atlas_controller = SimpleBoxAtlasController(com_pos_desired, K_p, K_d)
  return simple_box_atlas_controller
end


function compute_control_input(robot::BoxRobot, controller::SimpleBoxAtlasController, state::BoxRobotState, t::Float64, dt::Float64)
  """
  Computes the control input for this simple controller that drives com to com_pos_desired
    - assumes that both feet in contact, throw an error otherwise???

  Arguments:
    - t: the current simulation time
  Returns:
    - (BoxRobotInput, BoxRobotControllerData)
  """
  com_pos_error = controller.com_pos_desired - state.centroidal_dynamics_state.pos
  com_vel_error = - state.centroidal_dynamics_state.vel # corresponds to zero desired vel

  # desired com acceleration coming from PD controller
  com_acc_des = controller.K_p*com_pos_error + controller.K_d*com_vel_error

  # total force needed to achieve that desired com acceleration
  total_force = robot.mass*(com_acc_des - robot.gravity)

  # distribute that force between the two feet
  limb_vel = zeros(total_force)
  zero_force = zeros(total_force)
  limb_input_type = ConstantAccelerationLimbInput
  single_foot_force = total_force/2.0
  foot_limb_input = LimbInput(limb_vel, single_foot_force, true, limb_input_type)
  hand_limb_input = LimbInput(limb_vel, zero_force, false, limb_input_type)

  # move right hand towards wall at constant 1m/s
  # until it is in contact
  if !state.limb_states[:right_hand].in_contact
    right_hand_vel = [1.,0.]
    right_hand_input = LimbInput(right_hand_vel, zero_force, false, limb_input_type)
  else
    right_hand_input = LimbInput(limb_vel, zero_force, true, limb_input_type)
  end


  limb_inputs = Dict(:left_foot => foot_limb_input, :right_foot => foot_limb_input,
  :left_hand => hand_limb_input, :right_hand => right_hand_input)
  control_input = BoxRobotInput(limb_inputs)

  controller_data = SimpleBoxAtlasControllerData(t, com_acc_des)
  return control_input, controller_data
end


function input_with_no_force_at_distance!(state::BoxRobotState, input::BoxRobotInput)
  """
  Set has_force = false for limbs that aren't in contact yet
  This is a safety method, otherwise simulator will throw errors
  Returns:
    - modifies input inplace
  """

  for (limb_sym, limb_input) in input.limb_inputs
    if !state.limb_states[limb_sym].in_contact
      limb_input.has_force = false
    end
  end

end
