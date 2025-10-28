check_debug_log_out()

if(ENABLE_CPU)
  file(GLOB_RECURSE HARDWARE_CPU_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  add_library(hardware_cpu SHARED ${HARDWARE_CPU_SRC_FILES})
  target_link_libraries(hardware_cpu
      hardware_abstract
    )

  set_target_properties(hardware_cpu PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )

  install(TARGETS hardware_cpu
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)
endif()