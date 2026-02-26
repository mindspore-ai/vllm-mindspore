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

# TODO(jiaorui)
if(NOT EXISTS ${ASCEND_ATB_PATH})
    message(FATAL_ERROR "ATB not found at ${ASCEND_ATB_PATH}, stopping configuration")
endif()

file(GLOB_RECURSE OPS_ATB_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
include_directories(${ASCEND_ATB_PATH}/include)
find_library(ATB_KERNEL atb ${ASCEND_ATB_PATH}/lib)
add_library(ops_ascend_atb SHARED ${OPS_ATB_SRC_FILES})
target_link_libraries(ops_ascend_atb PRIVATE ${ATB_KERNEL}
        ops_ascend_aclnn ops_base_obj hardware_ascend kernel mrt::securec)

set_target_properties(ops_ascend_atb PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_atb
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)