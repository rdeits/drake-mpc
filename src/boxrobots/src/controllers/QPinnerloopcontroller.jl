using PyCall
@pyimport boxatlas
@pyimport boxatlas.boxatlas as boxatlastypes

abstract ContactSwitchType
immutable NoSwitch <: ContactSwitchType end
immutable MakingContact <: ContactSwitchType end
immutable BreakingContact <: ContactSwitchType end

type ContactSwitch{T<:ContactSwitchType}
  limb_symbol::Symbol
end

function ContactSwitch{T}(limb_symbol::Symbol, ::Type{T})
  return ContactSwitch{T}(limb_symbol)
end

function ContactSwitch(cs_0::ContactState, cs_1::ContactState)
  """
  Compute the contact switch type when transitioning from
  cs_0 --> cs_1
  """

  num_contact_state_mismatches = 0
  limb_symbol = Symbol()
  contact_switch_type = ContactSwitchType

  for (limb_sym, contact_indicator_0) in cs_0
    contact_indicator_1 = cs_1[limb_sym]
    # there was a contact switch
    if contact_indicator_0 != contact_indicator_1
      if (contact_indicator_0 & !contact_indicator_1)
        contact_switch_type = BreakingContact
      else
        contact_switch_type = MakingContact
      end
      limb_symbol = limb_sym
      num_contact_state_mismatches += 1
    end
  end

  if num_contact_state_mismatches > 1
    error("more than one state changed simultaneously, dont know how to handle this")
  end

  if num_contact_state_mismatches > 0
    contact_switch = ContactSwitch(limb_symbol, contact_switch_type)
    return true, contact_switch
  else
    return false, nothing
  end

end

function get_contact_switch_type{T}(contact_switch::ContactSwitch{T})
  return T
end


abstract ContactMismatchType
immutable NoMismatch <: ContactMismatchType end
immutable EarlyContact <: ContactMismatchType end
immutable LateContact <: ContactMismatchType end

type ContactMismatch{T<:ContactMismatchType}
  limb_sym::Symbol
end

function ContactMismatch{T}(limb_symbol::Symbol, ::Type{T})
  return ContactMismatch{T}(limb_symbol)
end

type QPInnerLoopControllerState
  t_start::Float64 # time at which soln_data was constructed
  soln_data::PyObject
  contact_plan::ContactPlan
  contact_state::ContactState
  contact_mismatch::ContactMismatch
end

function QPInnerLoopControllerState(t_start::Float64, state::BoxRobotState, soln_data::PyObject)
  contact_state = contact_state_from_robot_state(state)
  contact_plan = make_contact_plan(contact_state, soln_data)
  contact_mismatch = ContactMismatch(Symbol(), NoMismatch)
  return QPInnerLoopControllerState(t_start, soln_data, contact_plan, contact_state,
  contact_mismatch)
end

type QPInnerLoopController <: BoxRobotController
  python_controller::PyObject
  controller_state::QPInnerLoopControllerState
end

function QPInnerLoopController(python_controller::PyObject, t_start::Float64,
  state::BoxRobotState, soln_data::PyObject)
  controller_state = QPInnerLoopControllerState(t_start, state, soln_data)
  return QPInnerLoopController(python_controller, controller_state)
end

type QPInnerLoopControllerData <: BoxRobotControllerData{QPInnerLoopController}
  solve_time::Float64
  contact_assignments::Dict{Symbol, Vector}
  contact_mismatch::ContactMismatch
end


function get_contact_state_from_plan(t_plan::Float64, soln_data::PyObject)
  t_plan = clip_time_to_plan_limits(t_plan, soln_data)
  control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](
  t_plan, soln_data)
  contact_state = ContactState()
  for (limb_sym, limb_idx_python) in julia_to_python_limb_map
    contact_state[limb_sym] = control_input_python[:force_indicator][limb_idx_python]
  end
  return contact_state
end


function make_contact_plan(contact_state::ContactState, soln_data::PyObject)
  """
  # Arguments
  * `contact_state`: the current contact state

  """
  ts = soln_data[:ts]
  t_plan_end = soln_data[:ts][end] - 1e-3
  contact_state_vec = Vector{ContactState}()
  times = Vector{Float64}()
  push!(times, 0.)
  push!(contact_state_vec, get_contact_state_from_plan(0., soln_data))

  for t in ts[2:end]
    contact_state_next = get_contact_state_from_plan(t, soln_data)
    contact_state_current = contact_state_vec[end]
    has_contact_switch, contact_switch = ContactSwitch(contact_state_current, contact_state_next)
    if has_contact_switch
      push!(times, t)
      push!(contact_state_vec, contact_state_next)
    end
  end

  contact_plan_traj = Trajectory(times, contact_state_vec)


  # figure out next_contact_state_idx
  # if contact_state and contact_state_vec[1] are the SAME then
  # plan is starting from contact_state. The only way this wouldn't
  # be true is if the plan is breaking contact in the first timestep
  # then contact_state will have more cont
  next_contact_state_idx = 2 # default value
  has_contact_switch, contact_switch = ContactSwitch(contact_state, contact_state_vec[1])

  # need some special logic here if breaking contact on first
  if has_contact_switch
    next_contact_state_idx = 1
  end
  contact_plan = ContactPlan(contact_plan_traj, next_contact_state_idx)
end


# we need three versions of this function
function update_contact_mismatch_state!(t_plan::Float64, contact_state::ContactState,
  contact_plan::ContactPlan, contact_mismatch::ContactMismatch{NoMismatch})
  """
  Currently no contact mismatch. Could be at the beginning of a plan

  # Returns
  * next_contact_mismatch
  """
  planned_contact_state = contact_plan.plan(t_plan)
  nxt_contact_switch_time = get_next_contact_switch_time(contact_plan)


  if t_plan >= nxt_contact_switch_time
    print("\n\n-----------------------")
    print("-----------------------")
    print("\n\n----- t_plan > nxt_contact_switch_time --------\n\n")
    # could be making OR breaking contact

    # this should be the same as planned_contact_state in this case
    nxt_planned_contact_state = get_next_contact_state(contact_plan)
    has_contact_switch, contact_switch = ContactSwitch(contact_state, planned_contact_state)


    # some debugging print statements
    if false
      println("t_plan = ", t_plan)
      println("contact_state = ", contact_state)
      println("planned_contact_state = ", planned_contact_state)
      println("nxt_planned_contact_state = ", nxt_planned_contact_state)
    end

    if !has_contact_switch
      print("---- made contact switch on time ----")
      print("t_plan = ", t_plan)
      print("\n\n")
      # made contact in time
      if !isequal(nxt_planned_contact_state, contact_state)
        error("nxt_planned_contact_state not same as contact_state")
      end
      # increment the next_contact_state_idx counter
      increment_next_contact_state_idx!(contact_plan)
      return contact_mismatch
    end

    # if we got here then there is a mismatch between contact_state and
    # and planned_contact_state
    contact_switch_type = get_contact_switch_type(contact_switch)

    if contact_switch_type == MakingContact
      print("\n\n-----------------------")
      print("-----------------------")
      print("\n\n----------Late Contact----------\n")
      print("t_plan = ", t_plan)
      print("\n\n")
      print("\n\n-----------------------\n\n")
      nxt_contact_mismatch = ContactMismatch(contact_switch.limb_symbol, LateContact)
      increment_next_contact_state_idx!(contact_plan)
      return nxt_contact_mismatch
    else
      print("\n\n-----------------------")
      print("-----------------------")
      print("\n\n----------Breaking Contact----------\n")
      print("t_plan = ", t_plan)
      print("\n\n")
      print("\n\n-----------------------\n\n")

      # must be breaking contact, so we have no mismatch, just go with planned
      # contact assignments
      # Increment next_contact_state_idx in the contact_plan
      increment_next_contact_state_idx!(contact_plan)
      return contact_mismatch
    end
  else
    # could have NoMismatch or EarlyContact
    has_contact_switch = !isequal(contact_state, planned_contact_state)

    if has_contact_switch
      print("\n\n-----------------------\n\n")
      print("\n\n----------Early Contact-----------")
      print("t_plan = ", t_plan)
      print("\n\n")
      # we should be in EarlyContact here
      ~, contact_switch = ContactSwitch(planned_contact_state, contact_state)
      # determine the contact switch type
      contact_switch_type = get_contact_switch_type(contact_switch)

      # ensure that contact_state == nxt_planned_contact_state
      nxt_planned_contact_state = get_next_contact_state(contact_plan)

      # sanity check
      if !isequal(contact_state, nxt_planned_contact_state)
        error("had contact switch but next state is different from planned")
      end

      # sanity check
      if contact_switch_type != MakingContact
        error("in EarlyContact case but contact_switch_type isn't MakingContact")
      end

      # plan stuff will be updated when we switch from EarlyContact to normal contact
      nxt_contact_mismatch = ContactMismatch(contact_switch.limb_symbol, EarlyContact)
      return nxt_contact_mismatch
    else
      # nothing happened in this case
      return contact_mismatch
    end
  end
end

function update_contact_mismatch_state!(t_plan::Float64, contact_state::ContactState,
  contact_plan::ContactPlan, contact_mismatch::ContactMismatch{LateContact})
  """
  Check that we are in agreement with the planned contact state. If so then
  return a NoMismatch. Otherwise throw an error
  """
  planned_contact_state = get_contact_state_from_plan(t_plan)
  if !isequal(contact_state, planned_contact_state)
    error("In LateContact mode, but contact_states aren't equal")
  end

  return ContactMismatch(Symbol(), NoMismatch)
end

function update_contact_mismatch_state!(t_plan::Float64, contact_state::ContactState,
  contact_plan::ContactPlan, contact_mismatch::ContactMismatch{EarlyContact})
  """
  Checks to see if t_plan > nxt_contact_switch_time. If it is then switch to
  NoMismatch. If there is no next switch then throw an error
  """
  if !has_next_contact_switch(contact_plan)
    error("In state EarlyContact but plan has no more contact switches")
  end

  nxt_contact_switch_time = get_next_contact_switch_time(contact_plan)
  if (t_plan >= nxt_contact_switch_time)
    increment_next_contact_state_idx!(contact_plan)
    nxt_contact_mismatch = ContactMismatch(Symbol(), NoMismatch)
  else
    return contact_mismatch
  end
end

function extract_contact_assignments_from_plan(t_current::Float64, controller::QPInnerLoopController)
  """
  Gets the contact assigment from plan at knot point times specified by
  `t_plan, num_time_steps` and `dt`

  # Returns
  * `contact_assignments`: dict with contact assignment for each limb
  """

  soln_data = controller.controller_state.soln_data
  current_plan_time = t_current - controller.controller_state.t_start

  num_time_steps = controller.python_controller[:defaults]["num_time_steps"]
  dt = controller.python_controller[:defaults]["dt"]


  contact_assignments = Dict{Symbol, Vector}()
  for (limb_sym, _) in julia_to_python_limb_map
    contact_assignments[limb_sym] = Vector(num_time_steps)
  end

  for idx in 1:num_time_steps
    knot_point_plan_time = current_plan_time + (idx - 1)*dt

    # be careful that we don't index past the end of the plan
    # otherwise python Trajectory object gets angry
    knot_point_plan_time = clip_time_to_plan_limits(knot_point_plan_time, soln_data)
    control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](
    knot_point_plan_time, soln_data)

    for (limb_sym, limb_idx_python) in julia_to_python_limb_map
      contact_assignments[limb_sym][idx] = control_input_python[:force_indicator][limb_idx_python]
    end
  end

  return contact_assignments
end


function compute_contact_assignments(t::Float64, controller::QPInnerLoopController)
  contact_assignments = extract_contact_assignments_from_plan(t, controller)
  contact_mismatch = controller.controller_state.contact_mismatch
  adjust_contact_assignments!(contact_assignments, contact_mismatch)
  return contact_assignments
end

function adjust_contact_assignments!(contact_assignments,
  contact_mismatch::ContactMismatch{NoMismatch})
  """
  Don't need to do anything to planned contact assignments if there is no mismatch
  """
  return
end

function adjust_contact_assignments!(contact_assignments,
  contact_mismatch::ContactMismatch{EarlyContact})
  """
  Adjust planned contact sequence to account for early contact of limb specified
  in contact_mismatch. Set contact to true until plan has that limb being in contact

  Modifies contact_assignments in place
  """
  limb_sym = contact_mismatch.limb_sym
  limb_idx = julia_to_python_limb_map[limb_sym]

  contact_assignment = contact_assignments[limb_sym]
  for (idx, in_contact) in enumerate(contact_assignment)
    # set in_contact = true until plan says we are in contact
    if !in_contact
      contact_assignment[idx] = true
    else
      break
    end
  end
end

function adjust_contact_assignments!(contact_assignments,
  contact_mismatch::ContactMismatch{LateContact})
  """
  Adjust planned contact assignments to deal with LateContact.

  Specifically set current contact of that limb to false for current time step.
  This is simplest thing for now, can make it more complicated later as needed

  Modifies contact_assigments in place
  """
  limb_sym = contact_mismatch.limb_sym
  contact_assigments[limb_sym][1] = true
end

# function adjust_contact_assignments_old!(contact_assignments::Dict{Symbol, Vector}, contact_mismatch::Tuple)
#   limb_sym = contact_mismatch[2]
#   limb_contact_assignment = contact_assignments[limb_sym]
#
#
#   if contact_mismatch[1] == :early
#     for idx=1:length(limb_contact_assignment)
#      if limb_contact_assignment[idx]
#        break
#      else
#        limb_contact_assignment[idx] = true
#      end
#    end
#  elseif contact_mismatch[1] == :late
#    # push back contact by one, leave everything else the same
#    limb_contact_assignment[1] = false
#   else
#     error("contact_mismatch[1] must be either :early or :late")
#   end
#
#   return contact_assignments
# end

function convert_contact_assignments_to_python(contact_assignments::Dict{Symbol, Vector})
  """
  Converts the contact assignment into a the python form with limb_idx instead of
  limb_sym as the keys of the dict.
  """
  contact_assignments_python = Dict{Int64, Vector}()
  for (limb_sym, limb_contact_assignment) in contact_assignments
    # subtract 1 because python is 0 indexed
    limb_idx = julia_to_python_limb_map[limb_sym] - 1
    contact_assignments_python[limb_idx] = limb_contact_assignment
  end

  return contact_assignments_python
end

function compute_control_input(robot::BoxRobot, controller::QPInnerLoopController,
  state::BoxRobotState, t::Float64, dt::Float64)
  """
  Setup a QP using the mode sequence coming from the plan
  This changes the internal state of the controller
  """
  # get the contact asssignment from the plan
  controller_state = controller.controller_state
  controller_state.contact_state = contact_state_from_robot_state(state)
  t_plan = t - controller_state.t_start
  t_plan = clip_time_to_plan_limits(t_plan, controller.controller_state.soln_data)

  # compute next contact mismatch
  nxt_contact_mismatch = update_contact_mismatch_state!(t_plan,
  controller_state.contact_state,
  controller_state.contact_plan,
  controller_state.contact_mismatch)

  controller_state.contact_mismatch = nxt_contact_mismatch

  # get contact assignments from plan
  # this does the internal adjusting based on the contact_mismatch field of
  # controller_state
  contact_assignments = compute_contact_assignments(t, controller)
  contact_assignments_python = convert_contact_assignments_to_python(contact_assignments)

  # convert state to python and construct optimization problem
  python_controller = controller.python_controller
  robot_python = python_controller[:robot]
  state_python = convert_box_robot_state_to_python(robot_python, state)
  opt = python_controller[:construct_contact_stabilization_optimization](state_python,
  contact_assignments=contact_assignments_python)

  soln_data = pycall(opt[:solve], PyObject)
  t_plan = 0 # always extract first control input
  control_input_python = boxatlastypes.BoxAtlasInput[:get_input_from_soln_data](0,soln_data)

  # convert python results to correct julia types
  box_robot_input = convert_box_atlas_input_from_python(control_input_python)
  input_with_no_force_at_distance!(state, box_robot_input)
  contact_mismatch_copy = Base.deepcopy(controller_state.contact_mismatch)
  data = QPInnerLoopControllerData(soln_data[:solve_time], contact_assignments,
  contact_mismatch_copy)
  return box_robot_input, data
end
