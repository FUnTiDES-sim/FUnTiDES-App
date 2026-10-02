cmake_minimum_required(VERSION 3.22.1)

project(funtides_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_MPI "Build the driver with MPI domain decomposition" OFF)

# FUnTiDES brings Kokkos through its package configuration.
find_package(funtides REQUIRED)
find_package(ADIOS2 REQUIRED)
if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(funtides-sem
  src/main.cc
  src/sem_proxy.cc
)

target_include_directories(funtides-sem
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cxxopts
)

target_link_libraries(funtides-sem
  PRIVATE
    funtides::funtides
    adios2::adios2
)

if(USE_MPI)
  target_compile_definitions(funtides-sem PRIVATE USE_MPI)
  target_link_libraries(funtides-sem PRIVATE MPI::MPI_CXX)
endif()

install(TARGETS funtides-sem RUNTIME DESTINATION bin)

enable_testing()
add_subdirectory(tests/validation)
