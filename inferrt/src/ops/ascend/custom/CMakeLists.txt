check_debug_log_out()

file(GLOB OPS_CUSTOM_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_ascend_custom SHARED ${OPS_CUSTOM_SRC_FILES})

target_link_libraries(ops_ascend_custom PRIVATE
        ops_base_obj
        ops_utils
        hardware_ascend
        kernel
        mrt::securec
        ${CMAKE_DL_LIBS})

set_target_properties(ops_ascend_custom PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_custom
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)