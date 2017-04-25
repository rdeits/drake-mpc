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
env, robot = br.make_robot_and_environment(dist_to_left_wall=0.5)
robot_state = br.make_robot_state()
# robot_state.limb_states[:left_hand].pos[2] = 1.25
robot_state.centroidal_dynamics_state.vel[1] = -0.5

br.draw_environment(vis, env)
mass = robot.mass
gravity = robot.gravity
vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))


println("\n\n ----------- \n \n ")
println("Testing out python controller")
miqp_controller = br.MIQPController()
t = 0.0
dt = 0.05
duration = 1.0
robot_input, data = br.compute_control_input(robot, miqp_controller, robot_state, t, dt)
println("drawing robot state and input")
br.draw_box_robot_state(vis, robot_state; input=robot_input, options=vis_options)

@time traj = br.simulate_tspan(robot, miqp_controller, robot_state, dt, duration)
br.playback_trajectory(vis, traj; options=vis_options)


println("finished")
