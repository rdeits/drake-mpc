push!(LOAD_PATH, "../src/")
import ConvexMomentumDynamics
const CVM = ConvexMomentumDynamics

using DrakeVisualizer


println("Running CMD visualizer test")

# Launch the viewer application if it isn't running already:
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
vis = Visualizer()
delete!(vis) # clear all objects

# test drawing the COM
com_position = [0.,0.,1.]
# CVM.draw_com(vis, com_position)

# create a CentroidalDynamicsState struct
num_contacts = 2
contact_locations = Vector{Vector{Float64}}(num_contacts)
contact_locations[1] = [-0.5,0,0]
contact_locations[2] = [0.5,0,0]
display(contact_locations)

contact_forces = Vector{Vector{Float64}}(num_contacts)
contact_forces[1] = [0,0,0.5]
contact_forces[2] = [0,0,0.5]

state = CVM.CentroidalDynamicsState(com_position, contact_locations, contact_forces)
display(state)
# visualize the CDS
CVM.draw_centroidal_dynamics_state(vis, state)

println("finished CMD visualizer test")
