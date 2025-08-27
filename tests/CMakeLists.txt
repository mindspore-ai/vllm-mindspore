check_debug_log_out()

if(ENABLE_ASCEND)
  message(STATUS "Backend Ascend enabled")
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_PATH /usr/local/Ascend)
  endif()

  message("Note compile ascend path: ${ASCEND_PATH}")
  include_directories(${ASCEND_PATH}/latest/include/)
  link_directories(${ASCEND_PATH}/latest/lib64/)

  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  include_directories(${Python3_INCLUDE_DIRS})


  set(depname "pybind11")
  set(PYBIND11_PATH "${PROJECT_SOURCE_DIR}/${depname}-src")
  include_directories(${PYBIND11_PATH}/include)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")

  include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)

  add_executable(hardware_ascend_test_obj ./hardware/hardware_test.cc)
  target_link_libraries(hardware_ascend_test_obj hardware_ascend_obj)

  enable_testing()
  add_test(NAME hardware_ascend_test COMMAND hardware_ascend_test_obj)
else()
  message("Note ascend backend is disabled")
endif()
