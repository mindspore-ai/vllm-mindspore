check_debug_log_out()

file(GLOB_RECURSE HARDWARE_ABSTRACT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(hardware_abstract SHARED ${HARDWARE_ABSTRACT_SRC_FILES})
target_link_libraries(hardware_abstract dl)
