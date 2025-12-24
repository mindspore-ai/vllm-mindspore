check_debug_log_out()

file(GLOB_RECURSE OPS_BASE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_base_obj STATIC ${OPS_BASE_SRC_FILES})
target_link_libraries(ops_base_obj PRIVATE mrt_ir_obj ${TORCH_LIBRARIES})
if(ENABLE_ASCEND)
    target_compile_options(ops_base_obj PRIVATE -Wno-error=unused-function)
    target_link_libraries(ops_base_obj PRIVATE torch_npu)
endif()
