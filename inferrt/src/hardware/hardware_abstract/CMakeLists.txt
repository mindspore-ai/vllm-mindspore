check_debug_log_out()

file(GLOB_RECURSE HARDWARE_ABSTRACT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(hardware_abstract SHARED ${HARDWARE_ABSTRACT_SRC_FILES})
target_link_libraries(hardware_abstract mrt_common dl)

set_target_properties(hardware_abstract PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS hardware_abstract
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)