using PyCall
# need to call this from julia terminal after having sourced the drake-mpc
# environment file, this sets PYTHONPATH correctly
@pyimport boxatlas
@pyimport boxatlas.boxatlascontroller as boxatlascontroller
@pyimport boxatlas.boxatlas as boxatlastypes

type MIQPController <: BoxRobotController
  python_controller::PyObject
end

type MIQPControllerData <: BoxRobotControllerData{MIQPController}
  solve_time::Float64
end

# outer constructor for MIQPController
function MIQPController()
  python_controller = boxatlascontroller.BoxAtlasController()
  return MIQPController(python_controller)
end

function convert_box_robot_state_to_python{T}(robot_python, state::BoxRobotState{T})
  """
  Constructs a BoxAtlasState python object (from boxatlas.py) given a robot
  and a state
  """
  num_limbs = length(state.limb_states)
  qlimb = Vector{Vector{T}}(num_limbs)
  contact_indicator = Vector{Bool}(num_limbs)
  for (limb_sym, limb_state) in state.limb_states
    limb_idx = julia_to_python_limb_map[limb_sym]
    qlimb[limb_idx] = limb_state.pos
    contact_indicator[limb_idx] = limb_state.in_contact
  end

  qcom = state.centroidal_dynamics_state.pos
  vcom = state.centroidal_dynamics_state.vel
  state_python = boxatlastypes.BoxAtlasState(robot_python, qcom=qcom, vcom=vcom,
  qlimb=qlimb, contact_indicator=contact_indicator)

  return state_python
end

function compute_control_input(robot::BoxRobot, controller::MIQPController, state::BoxRobotState, t::Float64, dt::Float64)
  """
  Computes the control input by solving MIQP. Does this by calling python MIQP
  controller

  Arguments:
    - t: the current simulation time
  Returns:
    - (BoxRobotInput, BoxRobotControllerData)
  """
  # call into python to do the MIQP solve
  python_controller = controller.python_controller
  robot_python = python_controller[:robot]
  state_python = convert_box_robot_state_to_python(robot_python, state)
  opt = python_controller[:construct_contact_stabilization_optimization](state_python)
  soln_data = pycall(opt[:solve], PyObject)
  t_plan = 0 # always extract first control input
  control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](0,soln_data)

  # convert python results to correct julia types
  box_robot_input = convert_box_atlas_input_from_python(control_input_python)
  input_with_no_force_at_distance!(state, box_robot_input)
  data = MIQPControllerData(soln_data[:solve_time])
  return box_robot_input, data
end
