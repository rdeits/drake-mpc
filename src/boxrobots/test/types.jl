# function make_robot_and_environment()
#   floor_poly = br.polyhedron_from_bounds([-1,1],[-0.1,0])
#   floor = br.Surface(SimpleHRepresentation(floor_poly))
#
#   right_wall_poly = br.polyhedron_from_bounds([1,1.1],[-0.1,1.5])
#   right_wall = br.Surface(SimpleHRepresentation(right_wall_poly))
#
#   left_wall_poly = br.polyhedron_from_bounds([-1.1,-1],[-0.1,1.5])
#   left_wall = br.Surface(SimpleHRepresentation(left_wall_poly))
#
#   surfaces = [floor, right_wall, left_wall]
#   env = br.Environment(surfaces)
#
#   # make a robot
#   velocity_limit = 1.0
#   limb_bounds = br.h_representation_from_bounds([-1,1],[-1,1])
#   right_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)
#   left_foot = br.LimbConfig("right_foot", velocity_limit, limb_bounds, floor)
#
#   right_hand = br.LimbConfig("right_hand", velocity_limit, limb_bounds, right_wall)
#   left_hand = br.LimbConfig("left_hand", velocity_limit, limb_bounds, left_wall)
#
#   mass = 10.
#   dim = 2
#   gravity = [0.,-9.8]
#   limbs = Dict(:left_foot => left_foot, :right_foot => right_foot,
#   :left_hand => left_hand, :right_hand => right_hand)
#   robot = br.BoxRobot(mass, dim, gravity, limbs)
#   return env, robot
# end
#
# function make_robot_state()
#   # make a robot state
#   pos = [0.,1.]
#   vel = zeros(2)
#   centroidal_dynamics_state = br.CentroidalDynamicsState(pos, vel)
#
#   left_foot_pos = [-0.25, 0]
#   right_foot_pos = [0.25, 0]
#   left_foot_state = br.LimbState(left_foot_pos, vel, true)
#   right_foot_state = br.LimbState(right_foot_pos, vel, true)
#
#   left_hand_pos = [-0.35, 1.25]
#   right_hand_pos = [0.35, 1.25]
#   left_hand_state = br.LimbState(left_hand_pos, vel, false)
#   right_hand_state = br.LimbState(right_hand_pos, vel, false)
#
#   limb_states = Dict(:left_foot => left_foot_state, :right_foot => right_foot_state,
#   :left_hand => left_hand_state, :right_hand => right_hand_state)
#   robot_state = br.BoxRobotState(centroidal_dynamics_state, limb_states)
#   return robot_state
# end
#
# function make_robot_input()
#   mass = 10.
#   # construct robot input
#   vel = zeros(Float64, 2)
#   force = [0,mass*9.8/2.0]
#   limb_input_type = br.ConstantVelocityLimbInput
#   left_foot_input = br.LimbInput(vel, force, true, limb_input_type)
#   right_foot_input = br.LimbInput(vel, force, true, limb_input_type)
#
#
#   right_hand_vel = [1.,0.]
#   left_hand_input = br.LimbInput(vel, force, false, limb_input_type)
#   right_hand_input = br.LimbInput(right_hand_vel, force, false, limb_input_type)
#
#   limb_inputs = Dict(:left_foot => left_foot_input, :right_foot => right_foot_input,
#   :left_hand => left_hand_input, :right_hand => right_hand_input)
#   robot_input = br.BoxRobotInput(limb_inputs)
# end

@testset "construct robot and environment" begin
br.make_robot_and_environment()
end

@testset "make robot state" begin
br.make_robot_state()
end

@testset "make robot input" begin
  br.make_robot_input()
end
