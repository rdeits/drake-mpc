using DrakeVisualizer
import GeometryTypes: HyperRectangle, Vec, HomogenousMesh, Point
import ColorTypes: RGBA

@with_kw type BoxRobotVisualizerOptions
  force_arrow_normalizer::Float64 = 10.0
  com_radius::Float64 = 0.1
  contact_point_radius::Float64 = 0.05
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

  delete!(vis[:robot])

  println("drawing com")
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

  println("drawing com")
  com_position = convert_vector_to_3d(centroidal_dynamics_state.pos)
  p = make_point(com_position)
  sphere = HyperSphere(p, options.com_radius)
  println("made hypersphere")
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

function draw_force(vis::DrakeVisualizer.Visualizer, limb_sym::Symbol, limb_state::LimbState,
  limb_input::LimbInput, options::BoxRobotVisualizerOptions)
  contact_ray_start = convert_vector_to_3d(limb_state.pos)
  contact_ray_end = contact_ray_start +
  convert_vector_to_3d(limb_input.force)/options.force_arrow_normalizer

  force_line = PolyLine([contact_point.location, contact_ray_end], end_head=ArrowHead(0.05, 0.2), radius=0.02)
  grey = RGBA(1.,1.,1.)
  geometry_data = GeometryData(force_line, grey)
  setgeometry!(vis[:robot][limb_sym][:contact], geometry_data)
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
