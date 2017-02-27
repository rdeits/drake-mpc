type CentroidalDynamicsVisualizer
  com # DrakeVisualizer.Visualizer
  contacts # Array{DrakeVisualize.Visualizer, 1}
end

# encodes centroidal dynamics state at single instant
type CentroidalDynamicsState{T}
  com_position::Vector{T} # 3 vector
  contact_locations::Vector{Vector{T}} # 3 x num_contacts
  contact_forces::Vector{Vector{T}} # 3 x num_contacts
end

function make_point(position)
  return Point(position[1], position[2], position[3])
end

# create some drawing functionality
function draw_centroidal_dynamics_state(vis::DrakeVisualizer.Visualizer, state::CentroidalDynamicsState)
  """
  # Arguments
  * 'contact_locations': a 3 x num_contacts array specifying contact locations in world frame
  * 'contact_forces': a 3 x num_contacts array specifying contact forces in world frame

  """
  draw_com(vis, state.com_position)

  for i=1:length(state.contact_locations)
    draw_contact_point_and_force(vis, state.contact_locations[i], state.contact_forces[i])
  end
end

function draw_contact_point_and_force{T}(vis::DrakeVisualizer.Visualizer, contact_location::Vector{T}, contact_force::Vector{T})
  contact_ray_end = contact_location + contact_force
  force_line = PolyLine([contact_location, contact_ray_end], end_head=ArrowHead(0.05, 0.2), radius=0.02)
  # sphere = HyperSphere(make_point(contact_ray_end), 0.1)
  # contact_ray_end_vis = Visualizer(GeometryData(sphere, RGBA(1, 0, 0, 0.5)))
  setgeometry!(vis, GeometryData(force_line, RGBA(1, 0, 0, 0.5)))
end

function draw_com{T}(vis::DrakeVisualizer.Visualizer, com_position::Vector{T})
  sphere = HyperSphere(make_point(com_position), 0.1)
  setgeometry!(vis[:com], GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
  # Visualizer(GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
end
