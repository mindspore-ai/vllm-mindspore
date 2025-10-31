check_debug_log_out()

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH /usr/local/Ascend)
endif()

find_library(HCCL hccl
    PATHS ${ASCEND_PATH}/latest/lib64
    REQUIRED
    NO_DEFAULT_PATH
)

file(GLOB_RECURSE OPS_HCCL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(ops_ascend_hccl SHARED ${OPS_HCCL_SRC_FILES})
target_link_libraries(ops_ascend_hccl PRIVATE ops_base_obj hardware_ascend kernel mrt::securec ${HCCL})
