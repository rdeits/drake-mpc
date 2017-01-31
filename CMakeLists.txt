cmake_minimum_required(VERSION 3.5)
project(drake-mpc)

include(ExternalProject)

ExternalProject_Add(drake
  GIT_REPOSITORY git@github.com:rdeits/drake.git
  GIT_TAG 58af417fc2000a412f2c1427ee5d9cf8a2db755a
  GIT_SUBMODULES drake/doc/textbook
  CMAKE_ARGS
  	-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
)

add_subdirectory(src)

