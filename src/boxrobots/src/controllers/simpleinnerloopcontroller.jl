using PyCall
@pyimport boxatlas
@pyimport boxatlas.boxatlas as boxatlastypes

type SimpleInnerLoopController <: BoxRobotController
  t_start::Float64 # initial time at which python_soln_data was constructed
  soln_data::PyObject # boxatlas.BoxAtlasInput type from Python
end



# currently stores nothing
type SimpleInnerLoopControllerData <: BoxRobotControllerData{SimpleInnerLoopController}
  plan_time::Float64
  planned_input::BoxRobotInput
end

function compute_control_input(robot::BoxRobot, controller::SimpleInnerLoopController,
  state::BoxRobotState, t::Float64, dt::Float64)
  """
  Computes control input by indexing into plan with time t_plan = t - t_start
  Does heuristics on planned control input to make things feasible
  """

  # index into Python object to get the control data
  t_plan = t - controller.t_start
  # clip t_plan to limits, otherwise python Trajectory object gets mad
  t_plan = min(t_plan, controller.soln_data[:ts][end] - 1e-3)
  control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](t_plan,
  controller.soln_data)

  # TODO(manuelli): add heuristics for limb velocity when plan and state have mismatch
  box_robot_input = convert_box_atlas_input_from_python(control_input_python)
  planned_input = Base.deepcopy(box_robot_input)
  input_with_no_force_at_distance!(state, box_robot_input)
  controller_data = SimpleInnerLoopControllerData(t_plan, planned_input)
  return box_robot_input, controller_data
end
