check_debug_log_out()

add_subdirectory(utils)
file(GLOB_RECURSE OPS_ACLNN_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
list(FILTER OPS_ACLNN_SRC_FILES EXCLUDE REGEX "utils/.*")

set(AUTO_GEN_DIR ${TOP_DIR}/build/auto_gen)
set(OPS_ACLNN_AUTO_GEN_DIR ${AUTO_GEN_DIR}/ops/ascend/aclnn)
file(GLOB_RECURSE OPS_ACLNN_AUTO_GEN_FILES "${OPS_ACLNN_AUTO_GEN_DIR}/*.cc")
list(APPEND OPS_ACLNN_SRC_FILES ${OPS_ACLNN_AUTO_GEN_FILES})

add_library(ops_ascend_aclnn SHARED ${OPS_ACLNN_SRC_FILES})
include_directories(${AUTO_GEN_DIR})
target_link_libraries(ops_ascend_aclnn PRIVATE ops_ascend_aclnn_common ops_base_obj
                      hardware_ascend kernel mrt_profiler mrt::securec)

set_target_properties(ops_ascend_aclnn PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_aclnn
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
