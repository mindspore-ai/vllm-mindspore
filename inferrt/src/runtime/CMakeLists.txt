check_debug_log_out()

file(GLOB_RECURSE RUNTIME_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(runtime_obj STATIC ${RUNTIME_SRC_FILES})
target_link_libraries(runtime_obj ops_obj mrt_ir_obj kernel)

if(ENABLE_ASCEND)
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

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")
  target_link_libraries(runtime_obj hardware_ascend_obj)
endif()
