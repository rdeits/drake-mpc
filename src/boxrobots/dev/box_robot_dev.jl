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
mass = robot.mass
gravity = robot.gravity
vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))
vis = Visualizer()
delete!(vis)


env, robot = br.make_robot_and_environment()
robot_state = br.make_robot_state()

br.draw_environment(vis, env)

println("\n\n ----------- \n \n ")
println("Testing out python controller")
miqp_controller = br.MIQPController()
t = 0.0
dt = 0.05
robot_input = br.compute_control_input(robot, miqp_controller, robot_state, t, dt)
println("drawing robot state and input")
br.draw_box_robot_state(vis, robot_state; input=robot_input, options=vis_options)

println("finished")
