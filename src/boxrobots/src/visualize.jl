using DrakeVisualizer
import GeometryTypes: HyperRectangle, Vec, HomogenousMesh, Point
import ColorTypes: RGBA
using Interact, Reactive

@with_kw type BoxRobotVisualizerOptions
  force_arrow_normalizer::Float64 = 10.0
  com_radius::Float64 = 0.1
  contact_point_radius::Float64 = 0.05
  playback_dt::Float64 = 0.05
end


function draw_box_robot_state(vis::DrakeVisualizer.Visualizer, state::BoxRobotState;  options=nothing, input=nothing)
  """
  draws a box robot state
  Arguments:
    - options: BoxRobotVisualizerOptions
    - input: BoxRobotInput
  """

  # default for options
  if options == nothing
    options = BoxRobotVisualizerOptions()
  end

  # clear current robot drawing
  delete!(vis[:robot])
  draw_com(vis, state.centroidal_dynamics_state, options)

  # draw the limbs
  for v in state.limb_states
    limb_state = v[2]
    draw_limb(vis, v[1], limb_state, options)

    # only draw force if input is not nothing, AND there is force
    if input != nothing
      limb_input = input.limb_inputs[v[1]]
      if limb_input.has_force
        draw_contact_force(vis, v[1], limb_state, limb_input, options)
      end
    end
  end

end

function make_point(position)
  return Point(position[1], position[2], position[3])
end

function draw_com(vis::DrakeVisualizer.Visualizer, centroidal_dynamics_state::CentroidalDynamicsState,
  options::BoxRobotVisualizerOptions)

  com_position = convert_vector_to_3d(centroidal_dynamics_state.pos)
  p = make_point(com_position)
  sphere = HyperSphere(p, options.com_radius)
  setgeometry!(vis[:robot][:com][:position],GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
end

function draw_limb(vis::DrakeVisualizer.Visualizer, limb_sym::Symbol, limb_state::LimbState,
  options::BoxRobotVisualizerOptions)
  limb_position = convert_vector_to_3d(limb_state.pos)
  p = make_point(limb_position)
  sphere = HyperSphere(p, options.contact_point_radius)
  grey = RGBA(1.,1.,1.)
  setgeometry!(vis[:robot][limb_sym][:contact], GeometryData(sphere, grey))
end

function draw_contact_force(vis::DrakeVisualizer.Visualizer, limb_sym::Symbol, limb_state::LimbState,
  limb_input::LimbInput, options::BoxRobotVisualizerOptions)
  contact_ray_start = convert_vector_to_3d(limb_state.pos)
  contact_ray_end = contact_ray_start +
  convert_vector_to_3d(limb_input.force)/options.force_arrow_normalizer

  force_line = PolyLine([contact_ray_start, contact_ray_end], end_head=ArrowHead(0.05, 0.2), radius=0.02)
  grey = RGBA(1.,1.,1.)
  geometry_data = GeometryData(force_line, grey)
  setgeometry!(vis[:robot][limb_sym][:force], geometry_data)
end

function draw_environment(vis::DrakeVisualizer.Visualizer, env::Environment)
  """
  draws each surface specified in the environment
  """
  grey = RGBA(1.,1.,1.)
  for (idx, surface) in enumerate(env.surfaces)
    poly = polyhedron(surface.position, CDDLibrary())
    poly_3d = convert_polyhedron_to_3d(poly)
    setgeometry!(vis[:env][Symbol("surface$idx")], GeometryData(poly_3d, grey))
  end
end

function playback_trajectory(vis::DrakeVisualizer.Visualizer, traj::Trajectory{BoxRobotSimulationData}; options=nothing, playback_speed=1.0)
  """
  Draws each frame of the trajectory, sleeps for dt seconds in between draws
  Arguments:
    - playback_speed: 1.0 is 1X which is real time
  """

  # default for options
  if options == nothing
    options = BoxRobotVisualizerOptions()
  end

  t_start = traj.time[1]
  t_end = traj.time[end]
  t = copy(t_start)
  dt = options.playback_dt * playback_speed
  sleep_time = options.playback_dt
  # dt = traj.time[2] - traj.time[1]
  data = traj.data

  while (t <= t_end)
    d = eval(traj, t)
    state = d.state
    input = d.input
    draw_box_robot_state(vis::DrakeVisualizer.Visualizer, state::BoxRobotState;  options=options, input=input)
    sleep(sleep_time)
    t += dt
  end


  # for idx=1:num_time_steps
  #   t = t_start + (idx - 1)*dt
  #   d = eval(traj, t)
  #   state = d.state
  #   input = d.input
  #   draw_box_robot_state(vis::DrakeVisualizer.Visualizer, state::BoxRobotState;  options=options, input=input)
  #   sleep(dt)
  # end

end

function slider_playback(vis::DrakeVisualizer.Visualizer, traj::Trajectory{BoxRobotSimulationData}; options=nothing)
  """
  Constructs slider that can draw the trajectory
  """

  # default for options
  if options == nothing
    options = BoxRobotVisualizerOptions()
  end

  dt = traj.time[2] - traj.time[1]
  data = traj.data
  num_time_steps = length(data)

  @manipulate for idx=1:num_time_steps
    state = data[idx].state
    input = data[idx].input
    draw_box_robot_state(vis::DrakeVisualizer.Visualizer, state::BoxRobotState;  options=options, input=input)
  end
end
