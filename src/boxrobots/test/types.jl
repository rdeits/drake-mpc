@testset "construct robot and environment" begin
br.make_robot_and_environment()
end

@testset "make robot state" begin
br.make_robot_state()
end

@testset "make robot input" begin
  br.make_robot_input()
end
