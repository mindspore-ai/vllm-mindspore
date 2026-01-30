check_debug_log_out()

execute_process(COMMAND python -c "import pybind11; print(pybind11.__path__[0])"
        OUTPUT_VARIABLE PYBIND11_INSTALL_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYBIND11_INSTALL_PATH}")
find_package(pybind11 REQUIRED CONFIG)

find_package(Python3 COMPONENTS Development REQUIRED)

file(GLOB_RECURSE OPS_UTILS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_utils SHARED ${OPS_UTILS_SRC_FILES})
target_link_libraries(ops_utils PRIVATE mrt_common mrt_ir_obj ${CMAKE_DL_LIBS})
target_link_libraries(ops_utils PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(ops_utils PUBLIC Python3::Python)

if(ENABLE_ASCEND)
    add_compile_definitions(ENABLE_TORCH_NPU)
    include_directories(${TORCH_NPU_INCLUDE})
    target_link_directories(ops_utils PUBLIC ${TORCH_NPU_LIB_PATH})
    target_link_libraries(ops_utils PUBLIC torch_npu hardware_abstract)
endif()

set_target_properties(ops_utils PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_utils
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

