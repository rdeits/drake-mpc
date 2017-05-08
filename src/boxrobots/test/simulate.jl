@testset "simulate one step" begin
env, robot = br.make_robot_and_environment()
robot_state = br.make_robot_state()
robot_input = br.make_robot_input()
dt = 0.05
br.simulate(robot, robot_state, robot_input, dt)
end

@testset "simulate with simple controller" begin
# Simulate one step with controller
env, robot = br.make_robot_and_environment()
robot_state = br.make_robot_state()
robot_input = br.make_robot_input()

# make a simple controller
com_pos_desired = [0.,1.5]
K_p = 10.0
damping_ratio = 1.0
controller = br.simple_controller_from_damping_ratio(com_pos_desired, K_p, damping_ratio)

# simulate a timespan
duration = 5.0
dt = 0.05
@time data_array = br.simulate_tspan(robot, controller, robot_state, dt, duration)

# playback_trajectory
mass = robot.mass
gravity = robot.gravity
vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))

DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
vis = Visualizer()
delete!(vis)
br.draw_environment(vis,env)
br.playback_trajectory(vis, data_array; options=vis_options)

# think about putting some asserts in as to where the sim should end
end

@testset "simulate with QP controller" begin
# attempt to visualize a state
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
vis = Visualizer()
delete!(vis)
env, robot = br.make_robot_and_environment()
robot_state = br.make_robot_state()
robot_state.centroidal_dynamics_state.vel[1] = -1.75
br.draw_environment(vis, env)
mass = robot.mass
gravity = robot.gravity
vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))

miqp_controller = br.MIQPController()
t = 0.0
dt = 0.05
duration = 0.5
@time traj = br.simulate_tspan(robot, miqp_controller, robot_state, dt, duration)
br.playback_trajectory(vis, traj; options=vis_options)
end
