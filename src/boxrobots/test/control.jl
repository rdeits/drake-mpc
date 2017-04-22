@testset begin
  DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
  mass = robot.mass
  gravity = robot.gravity
  vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))
  vis = Visualizer()
  delete!(vis)

  env, robot = br.make_robot_and_environment()
  robot_state = br.make_robot_state()

  br.draw_environment(vis, env)

  miqp_controller = br.MIQPController()
  t = 0.0
  dt = 0.05
  robot_input = br.compute_control_input(robot, miqp_controller, robot_state, t, dt)
  br.draw_box_robot_state(vis, robot_state; input=robot_input, options=vis_options)
end
