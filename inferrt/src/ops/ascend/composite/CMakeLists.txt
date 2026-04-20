check_debug_log_out()

file(GLOB_RECURSE OPS_COMPOSITE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_ascend_composite SHARED ${OPS_COMPOSITE_SRC_FILES})
target_link_libraries(ops_ascend_composite PRIVATE
        ops_ascend_aclnn
        ops_base_obj
        hardware_ascend
        kernel
        mrt::securec)

set_target_properties(ops_ascend_composite PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_composite
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
