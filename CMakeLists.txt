cmake_minimum_required(VERSION 3.22.1)

project(funtides_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_MPI "Build funtides-sem with MPI domain decomposition" OFF)

# FUnTiDES brings Kokkos through its package configuration.
find_package(funtides REQUIRED)
find_package(Threads REQUIRED)
if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(examples)

enable_testing()
# Historical validation of funtides-sem against reference traces (normalized comparison).
if(TARGET funtides-sem)
  add_subdirectory(tests/validation)
endif()
# Every built method against the exact solution.
add_subdirectory(tests/analytical)
