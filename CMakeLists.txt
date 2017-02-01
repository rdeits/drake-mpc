cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:rdeits/drake.git
  GIT_TAG 561d276c008a300e8c269e0f7ee43d41ad759b7b
  GIT_SUBMODULES drake/doc/textbook
  CMAKE_CACHE_ARGS
  	-DWITH_GUROBI:bool=ON
  CMAKE_ARGS
  	-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  INSTALL_COMMAND ""
)

add_subdirectory(src)

configure_file(setup/setup.sh.in setup.sh @ONLY)
