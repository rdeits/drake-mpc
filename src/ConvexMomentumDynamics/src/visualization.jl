@with_kw type CentroidalDynamicsVisualizerOptions
  force_arrow_normalizer::Float64 = 10.0 # divide force by this to get arrow length
  playback_sleep_time::Float64 = 0.5
end

@with_kw type CentroidalDynamicsVisualizer
  p::CentroidalDynamicsOptimizationProblem
  vis::DrakeVisualizer.Visualizer = DrakeVisualizer.Visualizer()
  soln::OptimizationVariables{Float64}
  options::CentroidalDynamicsVisualizerOptions = CentroidalDynamicsVisualizerOptions()
end

# encodes centroidal dynamics state at single instant
immutable CentroidalDynamicsState{T}
  com_position::Vector{T} # 3 vector
  contact_points::Vector{ContactPoint} # 3 x num_contacts
  contact_forces::Vector{Vector{T}} # 3 x num_contacts
end

function get_centroidal_dynamics_state(p::CentroidalDynamicsOptimizationProblem, soln::OptimizationVariables{Float64}, knot_point::Int)
  # extract com position
  com_position = soln.com_position[:,knot_point]

  # parse the contact forces
  num_contacts = length(p.contact_points)
  contact_forces = Vector{Vector{Float64}}()
  for contact_idx=1:num_contacts
    push!(contact_forces, soln.forces[:,knot_point,contact_idx])
  end

  state = CentroidalDynamicsState(com_position, p.contact_points, contact_forces)
  return state
end

function make_point(position)
  return Point(position[1], position[2], position[3])
end

# create some drawing functionality
function draw_centroidal_dynamics_state(vis::DrakeVisualizer.Visualizer, state::CentroidalDynamicsState, options::CentroidalDynamicsVisualizerOptions)
  """
  Draws com location and contact forces
  """
  draw_com(vis, state.com_position)

  for i=1:length(state.contact_points)
    draw_contact_force(vis, state.contact_points[i], state.contact_forces[i], options)
  end
end

function draw_centroidal_dynamics_state(cd_vis::CentroidalDynamicsVisualizer, knot_point)
  state = get_centroidal_dynamics_state(cd_vis.p, cd_vis.soln, knot_point)
  draw_centroidal_dynamics_state(cd_vis.vis, state, cd_vis.options)
end

function draw_contact_force{T}(vis::DrakeVisualizer.Visualizer, contact_point::ContactPoint, contact_force::Vector{T}, options::CentroidalDynamicsVisualizerOptions)

  # scale the force by the parameter specified in options.
  contact_ray_end = contact_point.location + contact_force/options.force_arrow_normalizer
  force_line = PolyLine([contact_point.location, contact_ray_end], end_head=ArrowHead(0.05, 0.2), radius=0.02)
  geometry_data = GeometryData(force_line, RGBA(1, 0, 0, 0.5))
  setgeometry!(vis[:contact_points][Symbol(contact_point.name)], geometry_data)
end

function draw_com{T}(vis::DrakeVisualizer.Visualizer, com_position::Vector{T})
  sphere = HyperSphere(make_point(com_position), 0.1)
  setgeometry!(vis[:com], GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
end


function playback_trajectory(cd_vis::CentroidalDynamicsVisualizer)
  const num_timesteps = cd_vis.p.param.num_timesteps
  delete!(cd_vis.vis)
  for knot_point=1:num_timesteps
    draw_centroidal_dynamics_state(cd_vis, knot_point)
    sleep(cd_vis.options.playback_sleep_time)
  end
end
