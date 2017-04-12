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
poly = polyhedron_from_bounds([-1,1], [-1,1])
poly_3d = convert_polyhedron_to_3d(poly)
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

println("drawing environment")
br.draw_environment(vis,env)


# make a robot
velocity_limit = 1.0
limb_bounds = br.h_representation_from_bounds([-1,1],[-1,1])
right_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)
left_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)

mass = 10.
dim = 2
limbs = Dict(:left_foot => left_foot, :right_foot => right_foot)
robot = br.BoxRobot(mass, dim, limbs)


# make a robot state
pos = [0.,1.]
vel = zeros(2)
centroidal_dynamics_state = br.CentroidalDynamicsState(pos, vel)


left_foot_pos = [-0.25, 0]
right_foot_pos = [0.25, 0]
left_foot_state = br.LimbState(left_foot_pos, vel, false)
right_foot_state = br.LimbState(right_foot_pos, vel, false)

limb_states = Dict(:left_foot => left_foot_state, :right_foot => right_foot_state)
robot_state = br.BoxRobotState(centroidal_dynamics_state, limb_states)


# construct robot input
acceleration = zeros(Float64, 2)
force = [0,mass/2.0]
left_foot_input = br.LimbInput(acceleration, force, true)
right_foot_input = br.LimbInput(acceleration, force, true)

limb_inputs = Dict(:left_foot => left_foot_input, :right_foot => right_foot_input)
robot_input = br.BoxRobotInput(limb_inputs)

# visualize robot state
println("drawing robot state")
options = br.BoxRobotVisualizerOptions()
br.draw_box_robot_state(vis, robot_state)


println("drawing robot state and input")
br.draw_box_robot_state(vis, robot_state; input=robot_input)




println("finished")
