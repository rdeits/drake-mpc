cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:rdeits/drake.git
  GIT_TAG 5f9f2535ef496fc37d8c707451addf3d8d1e6342
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
