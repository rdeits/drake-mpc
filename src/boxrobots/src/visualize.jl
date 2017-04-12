using DrakeVisualizer
import GeometryTypes: HyperRectangle, Vec, HomogenousMesh, Point
import ColorTypes: RGBA

@with_kw type BoxRobotVisualizerOptions
  force_arrow_normalizer::Float64 = 10.0
  com_radius::Float64 = 0.1
  contact_point_radisu::Float64 = 0.025
end


function draw_box_robot_state(vis::DrakeVisualizer.Visualizer, state::BoxRobotState, input::BoxRobotInput, options::BoxRobotVisualizerOptions)
  """
  draws a box robot state
  """

  draw_com(vis, state.centroidal_dynamics_state, options)

  # draw the limbs
  for v in state.limb_state
    draw_limb(vis, v[2], options)
  end

  # draw the forces
  for v in input.limb_input
    draw_contact_force(vis, v[2], options)
  end
end

function draw_com(vis::DrakeVisualizer.Visualizer, centroidal_dynamics_state::CentroidalDynamicsState,
  options::BoxRobotVisualizerOptions)
  com_position = convert_vector_to_3d(centroidal_dynamics_state.position)
  p = Point(com_position)
  sphere = HyperSphere(point, options.com_radius)
  setgeometry!(vis[:com][:position],GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
end
