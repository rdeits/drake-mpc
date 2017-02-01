cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:rdeits/drake.git
  GIT_TAG 9d918dd2a18ed8dc0bccbaffd35b444e0ad8f36a
  GIT_SUBMODULES drake/doc/textbook
  CMAKE_CACHE_ARGS
  	-DWITH_GUROBI:bool=ON
  CMAKE_ARGS
  	-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  INSTALL_COMMAND ""
)

add_subdirectory(src)

configure_file(setup/setup.sh.in setup.sh @ONLY)
