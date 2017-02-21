cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:RobotLocomotion/drake.git
  GIT_TAG 4ba9eb2119562adf674e80a6e7489e29d0b6a40a
  GIT_SUBMODULES drake/doc/textbook
  CMAKE_CACHE_ARGS
  	-DWITH_GUROBI:bool=ON
  	-DWITH_MOSEK:bool=ON
  	-DWITH_IRIS:bool=ON
  CMAKE_ARGS
  	-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  INSTALL_COMMAND ""
)

add_subdirectory(src)

configure_file(setup/setup.sh.in setup.sh @ONLY)
