# Acoustic examples of every built method against the exact homogeneous solution, without
# normalization. Cases and tolerances follow the FUnTiDES Python validation (tests/validation/cases.py):
# relative L2 thresholds about 2 to 4 times the error measured there.
add_executable(validate_analytical validate_analytical.cc)
target_link_libraries(validate_analytical PRIVATE funtides::funtides)
target_compile_definitions(validate_analytical PRIVATE FUNTIDES_BIN_DIR="${CMAKE_BINARY_DIR}/bin")

set(ANALYTICAL_TOLERANCE_sem_1 3e-2)
set(ANALYTICAL_TOLERANCE_sem_2 3e-3)
set(ANALYTICAL_TOLERANCE_sem_3 3e-3)
set(ANALYTICAL_TOLERANCE_dg_1 5e-2)
set(ANALYTICAL_TOLERANCE_dg_2 2e-3)
set(ANALYTICAL_TOLERANCE_dg_3 2e-3)
# p-adaptive, per (order-min, order): the mesh is sized for the high order, so the under-resolved
# low-order part dominates the error.
set(ANALYTICAL_TOLERANCE_padaptive_1_2 7e-2)
set(ANALYTICAL_TOLERANCE_padaptive_1_3 1.2e-1)
set(ANALYTICAL_TOLERANCE_padaptive_2_3 3e-3)

# Each test runs in its own directory so that tests can run in parallel.
function(add_analytical_test method mesh order order_min tolerance)
  if(order_min GREATER 0)
    set(name analytical_${method}_order${order_min}to${order}_${mesh})
  else()
    set(name analytical_${method}_order${order}_${mesh})
  endif()
  set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/${name})
  file(MAKE_DIRECTORY ${work_dir})
  set(args --method ${method} --mesh ${mesh} --order ${order} --tolerance ${tolerance})
  if(order_min GREATER 0)
    list(APPEND args --order-min ${order_min})
  endif()
  add_test(NAME ${name} COMMAND validate_analytical ${args} WORKING_DIRECTORY ${work_dir})
  set_tests_properties(${name} PROPERTIES LABELS "analytical;analytical_${method};mesh_${mesh}" TIMEOUT 1800)
endfunction()

set(ANALYTICAL_DEPENDENCIES validate_analytical)
foreach(METHOD sem dg dg-sem dg-padaptive)
  if(TARGET funtides-${METHOD})
    list(APPEND ANALYTICAL_DEPENDENCIES funtides-${METHOD})
  endif()
endforeach()

foreach(MESH cartesian ucartesian)
  foreach(ORDER 1 2 3)
    if(TARGET funtides-sem)
      add_analytical_test(sem ${MESH} ${ORDER} 0 ${ANALYTICAL_TOLERANCE_sem_${ORDER}})
    endif()
    if(TARGET funtides-dg)
      add_analytical_test(dg ${MESH} ${ORDER} 0 ${ANALYTICAL_TOLERANCE_dg_${ORDER}})
    endif()
    if(TARGET funtides-dg-sem)
      add_analytical_test(dg-sem ${MESH} ${ORDER} 0 ${ANALYTICAL_TOLERANCE_dg_${ORDER}})
    endif()
  endforeach()
  if(TARGET funtides-dg-padaptive)
    foreach(PAIR 1_2 1_3 2_3)
      string(REPLACE "_" ";" ORDERS ${PAIR})
      list(GET ORDERS 0 ORDER_MIN)
      list(GET ORDERS 1 ORDER_MAX)
      add_analytical_test(dg-padaptive ${MESH} ${ORDER_MAX} ${ORDER_MIN} ${ANALYTICAL_TOLERANCE_padaptive_${PAIR}})
    endforeach()
  endif()
endforeach()

# make validate_methods: every analytical test; one method with ctest -L analytical_<method>.
add_custom_target(validate_methods
  COMMAND ${CMAKE_CTEST_COMMAND} -L analytical --output-on-failure
  DEPENDS ${ANALYTICAL_DEPENDENCIES}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running the analytical validation of every built method"
)
