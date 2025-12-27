check_debug_log_out()

# pybind11
execute_process(COMMAND python -c "import pybind11; print(pybind11.__path__[0])"
        OUTPUT_VARIABLE PYBIND11_INSTALL_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYBIND11_INSTALL_PATH}")
find_package(pybind11 REQUIRED CONFIG)

file(GLOB OPS_CUSTOM_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_ascend_custom SHARED ${OPS_CUSTOM_SRC_FILES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ops_ascend_custom PRIVATE
            -Wno-error=unused-function
            -Wno-unused-function
    )
endif()

find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${TORCH_NPU_INCLUDE})
target_link_libraries(ops_ascend_custom PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(ops_ascend_custom PUBLIC Python3::Python)
target_link_directories(ops_ascend_custom PUBLIC ${TORCH_NPU_LIB_PATH})
target_link_libraries(ops_ascend_custom PUBLIC torch_npu)


target_link_libraries(ops_ascend_custom PRIVATE
        ops_base_obj
        ops_utils
        hardware_ascend
        kernel
        mrt::securec
        ${CMAKE_DL_LIBS}
        pybind11::module
        pybind11::embed)

set_target_properties(ops_ascend_custom PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_custom
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)