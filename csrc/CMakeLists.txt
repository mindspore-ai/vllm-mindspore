cmake_minimum_required(VERSION 3.16)
project(Ops)

set(MS_EXTENSION_NAME "" CACHE STRING "Extension Name")
set(BUILD_EXTENSION_DIR "" CACHE STRING "Extension directory")
if (MS_EXTENSION_NAME STREQUAL "")
    message(FATAL_ERROR "MS_EXTENSION_NAME must be set. Use -DMS_EXTENSION_NAME=<name>")
endif()
if (BUILD_EXTENSION_DIR STREQUAL "")
    message(FATAL_ERROR "BUILD_EXTENSION_DIR must be set. Use -DBUILD_EXTENSION_DIR=<path>")
endif()

# Build ascendc kernels
add_subdirectory(ascendc)

# Collect source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/module/*.cpp)

# Generate a temporary python script file to build custom ops with MindSpore's CustomOpBuilder
set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms
src_files = '${SRC_FILES}'.split(';')
ms.ops.CustomOpBuilder(
    name='${MS_EXTENSION_NAME}',
    sources=src_files,
    backend='Ascend',
    cflags='-I${CMAKE_CURRENT_SOURCE_DIR}',
    ldflags='-L${ASCENDC_TARGET_DIR} -l${ASCENDC_TARGET_NAME}',
    build_dir='${BUILD_EXTENSION_DIR}'
).build()
")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    DEPENDS ${ASCENDC_TARGET_NAME}
    COMMENT "Building custom operator with MindSpore"
)
