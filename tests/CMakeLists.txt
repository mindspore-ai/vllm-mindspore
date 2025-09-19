check_debug_log_out()

if(ENABLE_ASCEND AND BUILD_TESTS)
  message(STATUS "Backend Ascend enabled")
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_PATH /usr/local/Ascend)
  endif()

  message("Note compile ascend path: ${ASCEND_PATH}")
  include_directories(${ASCEND_PATH}/latest/include/)
  link_directories(${ASCEND_PATH}/latest/lib64/)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")

  include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)

  add_executable(run_hardware_ascend_test ./hardware/hardware_test.cc)
  target_link_libraries(run_hardware_ascend_test hardware_ascend)

  enable_testing()
  add_test(NAME hardware_ascend_test COMMAND run_hardware_ascend_test)
else()
  message("Note ascend backend is disabled")
endif()


if(ENABLE_CPU AND BUILD_TESTS)
  message(STATUS "Backend CPU enabled")
  include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)
  add_executable(run_hardware_cpu_test ./hardware/hardware_cpu_test.cc)
  target_link_libraries(run_hardware_cpu_test hardware_cpu)
  enable_testing()
  add_test(NAME hardware_cpu_test COMMAND run_hardware_cpu_test)
else()
  message("Note cpu backend is disabled")
endif()