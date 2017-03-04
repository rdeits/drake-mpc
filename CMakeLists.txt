cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

option(WITH_GUROBI "build with Gurobi" ON)
option(WITH_MOSEK "build with Mosek" ON)
option(WITH_IRIS "build with IRIS" ON)
option(WITH_DIRECTOR "build with Director" ON)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:RobotLocomotion/drake.git
  GIT_TAG d94ab70c5
  GIT_SUBMODULES drake/doc/textbook
  CMAKE_CACHE_ARGS
  	-DWITH_GUROBI:bool=${WITH_GUROBI}
  	-DWITH_MOSEK:bool=${WITH_MOSEK}
  	-DWITH_IRIS:bool=${WITH_IRIS}
    -DWITH_DIRECTOR:bool=${WITH_DIRECTOR}
    -DWITH_BULLET:bool=OFF
    -DWITH_CCD:bool=OFF
    -DWITH_FCL:bool=OFF
    -DWITH_HSRB_DESCRIPTION:bool=OFF
    -DWITH_IPOPT:bool=OFF
    -DWITH_NLOPT:bool=OFF
    -DWITH_OCTOMAP:bool=OFF
    -DWITH_SEDUMI:bool=OFF
    -DWITH_SPOTLESS:bool=OFF
  CMAKE_ARGS
  	-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  INSTALL_COMMAND ""
)

# add_subdirectory(src)

configure_file(setup/setup.sh.in setup.sh @ONLY)
