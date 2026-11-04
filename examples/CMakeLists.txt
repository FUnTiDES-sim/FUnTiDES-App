# One example executable per numerical method. A method is built only if the
# installed FUnTiDES was compiled with it (COMPILE_SEM, COMPILE_DG, ...).
get_target_property(FUNTIDES_SOLVER_DEFINITIONS funtides::solver INTERFACE_COMPILE_DEFINITIONS)

add_library(funtides_examples_common STATIC
  common/example_acquisition.cc
  common/example_mesh.cc
  common/example_output.cc
)
target_include_directories(funtides_examples_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/external/cxxopts
)
target_link_libraries(funtides_examples_common PUBLIC funtides::funtides Threads::Threads)

# funtides_add_example(<executable> <source> <FUnTiDES method definition>)
function(funtides_add_example name source method_definition)
  if(NOT "${method_definition}" IN_LIST FUNTIDES_SOLVER_DEFINITIONS)
    message(STATUS "${name} not built: FUnTiDES was built without ${method_definition}")
    return()
  endif()
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE funtides_examples_common)
  install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

funtides_add_example(funtides-sem sem/sem_example.cc COMPILE_SEM)
funtides_add_example(funtides-dg dg/dg_example.cc COMPILE_DG)
funtides_add_example(funtides-dg-sem dg_sem/dg_sem_example.cc COMPILE_DG_SEM)
funtides_add_example(funtides-dg-padaptive dg_padaptive/dg_padaptive_example.cc COMPILE_DG_PADAPTIVE)

if(USE_MPI AND TARGET funtides-sem)
  target_compile_definitions(funtides-sem PRIVATE USE_MPI)
  target_link_libraries(funtides-sem PRIVATE MPI::MPI_CXX)
endif()
