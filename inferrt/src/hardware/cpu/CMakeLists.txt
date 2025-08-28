check_debug_log_out()

if(ENABLE_CPU)
  file(GLOB_RECURSE HARDWARE_CPU_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  add_library(hardware_cpu_obj SHARED ${HARDWARE_CPU_SRC_FILES})
  target_link_libraries(hardware_cpu_obj
      hardware_abstract_obj
    )
endif()
