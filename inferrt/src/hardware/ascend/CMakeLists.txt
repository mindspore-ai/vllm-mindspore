check_debug_log_out()

if(ENABLE_ASCEND)
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_PATH /usr/local/Ascend)
  endif()

  message("Note compile ascend path: ${ASCEND_PATH}")
  include_directories(${ASCEND_PATH}/latest/include/)
  include_directories(${ASCEND_PATH}/latest/lib64/)
  find_library(ASCENDCL_LIB ascendcl
      PATHS ${ASCEND_PATH}/latest/lib64
      REQUIRED
      NO_DEFAULT_PATH
    )
  find_library(RUNTIME_LIB runtime
    PATHS ${ASCEND_PATH}/latest/lib64
    REQUIRED
    NO_DEFAULT_PATH
  )
  link_directories(${ASCEND_PATH}/latest/lib64/)

  file(GLOB_RECURSE HARDWARE_ASCEND_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  add_library(hardware_ascend_obj SHARED ${HARDWARE_ASCEND_SRC_FILES})
  target_link_libraries(hardware_ascend_obj
      hardware_abstract_obj
      ${ASCENDCL_LIB}
      ${RUNTIME_LIB}
    )
endif()
