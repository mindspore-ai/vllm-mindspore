check_debug_log_out()

# pybind11
execute_process(COMMAND python -c "import pybind11; print(pybind11.__path__[0])"
        OUTPUT_VARIABLE PYBIND11_INSTALL_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYBIND11_INSTALL_PATH}")
find_package(pybind11 REQUIRED CONFIG)

file(GLOB_RECURSE OPS_BASE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_base_obj STATIC ${OPS_BASE_SRC_FILES})
target_link_libraries(ops_base_obj PRIVATE mrt_ir_obj ${TORCH_LIBRARIES})
target_link_libraries(ops_base_obj PRIVATE ops_utils  pybind11::module pybind11::embed)
if(ENABLE_ASCEND)
    add_compile_definitions(ENABLE_TORCH_NPU)
    target_compile_options(ops_base_obj PRIVATE -Wno-error=unused-function)
    target_link_libraries(ops_base_obj PRIVATE torch_npu)
endif()
