check_debug_log_out()

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH_BASE $ENV{ASCEND_CUSTOM_PATH})
    if(ASCEND_PATH_BASE MATCHES "(cann|ascend-toolkit)$")
        get_filename_component(ASCEND_PATH_BASE ${ASCEND_PATH_BASE} DIRECTORY)
    endif()
else()
    set(ASCEND_PATH_BASE /usr/local/Ascend)
endif()

set(ASCEND_ATB_PATH ${ASCEND_PATH_BASE}/nnal/atb/latest/atb/cxx_abi_0)
if(EXISTS ${ASCEND_ATB_PATH})
    include_directories(${ASCEND_ATB_PATH}/include)
endif()

file(GLOB_RECURSE OPS_COMPOSITE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_ascend_composite SHARED ${OPS_COMPOSITE_SRC_FILES})
target_link_libraries(ops_ascend_composite PRIVATE
        ops_ascend_aclnn
        ops_ascend_atb
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