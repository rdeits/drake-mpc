@testset "draw robot state" begin
  env, robot = br.make_robot_and_environment()
  robot_state = br.make_robot_state()
  robot_input = br.make_robot_input()
  mass = robot.mass
  gravity = robot.gravity
  vis_options = br.BoxRobotVisualizerOptions(force_arrow_normalizer=mass*abs(gravity[end]))

  DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
  println("waiting for visualizer to start up")
  sleep(5) # let visualizer start up

  vis = Visualizer()
  delete!(vis)
  br.draw_environment(vis,env)
  br.draw_box_robot_state(vis, robot_state; input=robot_input, options=vis_options)
end
