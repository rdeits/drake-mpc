path_to_my_module = string(Base.source_dir(), "/../src/")
push!(LOAD_PATH, path_to_my_module)

reload("BoxRobots")
import BoxRobots
using DrakeVisualizer
import ColorTypes: RGBA
using Polyhedra: SimpleHRepresentation

br = BoxRobots
# attempt to visualize a state
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
vis = Visualizer()
delete!(vis)

# test constructing and drawing a polyhedron
poly = br.polyhedron_from_bounds([-1,1], [-1,1])
poly_3d = br.convert_polyhedron_to_3d(poly)
green = RGBA(0.,1.0,0.0)
# setgeometry!(vis[:poly], GeometryData(poly_3d, green))

# Construct an environemnt
println("constructing a surface")
floor_poly = br.polyhedron_from_bounds([-1,1],[-0.1,0])
floor = br.Surface(SimpleHRepresentation(floor_poly))

right_wall_poly = br.polyhedron_from_bounds([1,1.1],[-0.1,1.5])
right_wall = br.Surface(SimpleHRepresentation(right_wall_poly))

left_wall_poly = br.polyhedron_from_bounds([-1.1,-1],[-0.1,1.5])
left_wall = br.Surface(SimpleHRepresentation(left_wall_poly))

surfaces = [floor, right_wall, left_wall]
env = br.Environment(surfaces)
println("typeof(left_wall) = ", typeof(left_wall))

println("drawing environment")
br.draw_environment(vis,env)


# make a robot
velocity_limit = 1.0
limb_bounds = br.h_representation_from_bounds([-1,1],[-1,1])
right_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)
left_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)

right_hand = br.LimbConfig("right_hand", velocity_limit, limb_bounds, right_wall)
left_hand = br.LimbConfig("left_hand", velocity_limit, limb_bounds, left_wall)

mass = 10.
dim = 2
gravity = [0.,-9.8]
limbs = Dict(:left_foot => left_foot, :right_foot => right_foot,
:left_hand => left_hand, :right_hand => right_hand)
robot = br.BoxRobot(mass, dim, gravity, limbs)
vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))

# make a robot state
pos = [0.,1.]
vel = zeros(2)
centroidal_dynamics_state = br.CentroidalDynamicsState(pos, vel)


left_foot_pos = [-0.25, 0]
right_foot_pos = [0.25, 0]
left_foot_state = br.LimbState(left_foot_pos, vel, true)
right_foot_state = br.LimbState(right_foot_pos, vel, true)

left_hand_pos = [-0.35, 1.25]
right_hand_pos = [0.35, 1.25]
left_hand_state = br.LimbState(left_hand_pos, vel, false)
right_hand_state = br.LimbState(right_hand_pos, vel, false)

limb_states = Dict(:left_foot => left_foot_state, :right_foot => right_foot_state,
:left_hand => left_hand_state, :right_hand => right_hand_state)
robot_state = br.BoxRobotState(centroidal_dynamics_state, limb_states)


# construct robot input
vel = zeros(Float64, 2)
force = [0,mass*9.8/2.0]
limb_input_type = br.ConstantVelocityLimbInput
left_foot_input = br.LimbInput(vel, force, true, limb_input_type)
right_foot_input = br.LimbInput(vel, force, true, limb_input_type)


right_hand_vel = [1.,0.]
left_hand_input = br.LimbInput(vel, force, false, limb_input_type)
right_hand_input = br.LimbInput(right_hand_vel, force, false, limb_input_type)

limb_inputs = Dict(:left_foot => left_foot_input, :right_foot => right_foot_input,
:left_hand => left_hand_input, :right_hand => right_hand_input)
robot_input = br.BoxRobotInput(limb_inputs)

# visualize robot state
println("drawing robot state")
options = br.BoxRobotVisualizerOptions()
br.draw_box_robot_state(vis, robot_state, options=vis_options)


println("drawing robot state and input")
br.draw_box_robot_state(vis, robot_state; input=robot_input, options=vis_options)

## Simulate one step
print("simulating one step")
println(typeof(robot))
dt = 0.05
num_time_steps = 30
robot_state_final = robot_state
for i = 1:num_time_steps
  robot_state_final = br.simulate(robot, robot_state_final, robot_input, dt)
  # if robot_state_final.limb_states[:right_hand].in_contact
  #   println("i = ", i)
  #   break
  # end
end


br.draw_box_robot_state(vis, robot_state_final)
pos_next = robot_state_final.centroidal_dynamics_state.pos
println("simulation results \n \n")
println("pos= ", pos_next)
println("left_foot.pos ", robot_state_final.limb_states[:left_foot].pos)
println("left_foot.vel ", robot_state_final.limb_states[:left_foot].vel)
println("left_foot.in_contact ", robot_state_final.limb_states[:left_foot].in_contact)
println("right_hand.pos ", robot_state_final.limb_states[:right_hand].pos)
println("right_hand.in_contact ", robot_state_final.limb_states[:right_hand].in_contact)



# Simulate one step with controller
println("simulating robot + controller \n \n ")
com_pos_desired = [0.,1.5]
K_p = 10.0
damping_ratio = 1.0
controller = br.simple_controller_from_damping_ratio(com_pos_desired, K_p, damping_ratio)

# simulate a timespan
duration = 5.0
dt = 0.05
@time data_array = br.simulate_tspan(robot, controller, robot_state, dt, duration)
println("\n\n")

br.playback_trajectory(vis, data_array; options=vis_options)


idx = 5
t = data_array.tBreaks[idx]
data = data_array.data[idx]
println("t = ", t)
println("com_pos ", data.state.centroidal_dynamics_state.pos)
println("com_vel ", data.state.centroidal_dynamics_state.vel)

println("com_acc ", data.controller_data.com_acc)

println("finished")
