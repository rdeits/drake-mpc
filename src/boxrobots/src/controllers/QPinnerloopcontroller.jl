using PyCall
@pyimport boxatlas
@pyimport boxatlas.boxatlas as boxatlastypes

type QPInnerLoopController <: BoxRobotController
  t_start::Float64 # initial time at which soln_data was constructed
  soln_data::PyObject # SolutionData namedtuple from Python
  prev_state::BoxRobotState # need this for computing whether we are making/breaking contact
  num_time_steps::Int # number of time steps in the optimization
  dt::Float64 # the dt in the optimization
end

type QPInnerLoopControllerData <: BoxRobotControllerData{QPInnerLoopController}
  solve_time::Float64
end

function compute_control_input(robot::BoxRobot, controller::QPInnerLoopController,
  state::BoxRobotState, t::Float64, dt::Float64)
  """
  Setup a QP using the mode sequence coming from the plan
  """

  compute_contact_assigment(t, state, controller)
end

function extract_contact_assignment_from_plan(t_current::Float64, controller::QPInnerLoopController)
  """
  Gets the contact assigment from plan at knot point times specified by
  `t_plan, num_time_steps` and `dt`

  # Returns
  * `contact_assignment`: dict with contact assignment for each limb
  """

  t_plan_end = controller.soln_data[:ts][end] - 1e-3
  current_plan_time = t_current - controller.t_start
  num_time_steps = controller.num_time_steps

  contact_assignment = Dict{Symbol, Vector}()
  for (limb_sym, _) in julia_to_python_limb_map
    contact_assignment[limb_sym] = Vector(num_time_steps)
  end

  for idx in 1:num_time_steps
    knot_point_plan_time = current_plan_time + (idx - 1)*controller.dt

    # be careful that we don't index past the end of the plan
    # otherwise python Trajectory object gets angry
    knot_point_plan_time = min(knot_point_plan_time, t_plan_end)
    control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](
    knot_point_plan_time, controller.soln_data)

    for (limb_sym, limb_idx_python) in julia_to_python_limb_map
      contact_assignment[limb_sym][idx] = control_input_python[:force_indicator][limb_idx_python]
    end
  end

  return contact_assignment
end

function compute_contact_assignment(t::Float64, state::BoxRobotState, controller::QPInnerLoopController)
  """
  Compute the contact assignment that will be used for the qp
  """
  # extract planned contact assignment
  contact_assignment = extract_contact_assignment_from_plan(t, controller)

  num_contact_state_mismatches = 0
  contact_mismatch = tuple()

  for (limb_sym, limb_state) in state.limb_states
    planned_force_indicator = contact_assignment[limb_sym][1]
    if limb_state.in_contact & !planned_force_indicator
      # early contact
      println("early contact")
      contact_mismatch = (:early, limb_sym)
      num_contact_state_mismatches += 1
    elseif !limb_state.in_contact & planned_force_indicator
      # late contact
      println("late contact")
      num_contact_state_mismatches += 1
      contact_mismatch = (:late, limb_sym)
    end
  end

  # if there were contact mismatches then adjust the contact_assignment
  # accordingly
  if num_contact_state_mismatches > 1
    error("num_contact_state_mismatches greater than 1, don't know how to handle
    this situation")
  end

  if num_contact_state_mismatches > 0
    adjust_contact_assignment!(contact_assignment, contact_mismatch)
  end

  return contact_assignment
end


function adjust_contact_assignment!(contact_assignment::Dict{Symbol, Vector}, contact_mismatch::Tuple)
  limb_sym = contact_mismatch[2]
  limb_contact_assignment = contact_assignment[limb_sym]


  if contact_mismatch[1] == :early
    for idx=1:length(limb_contact_assignment)
     if limb_contact_assignment[idx]
       break
     else
       limb_contact_assignment[idx] = true
     end
   end
 elseif contact_mismatch[1] == :late
   # push back contact by one, leave everything else the same
   limb_contact_assignment[1] = false
  else
    error("contact_mismatch[1] must be either :early or :late")
  end

  return contact_assignment
end
