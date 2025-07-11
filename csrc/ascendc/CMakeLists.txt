cmake_minimum_required(VERSION 3.16)
project(AscendC_Kernels)

# Parameters passed from command line or default values
set(RUN_MODE "npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "system on chip type")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug")

# Set ASCEND_CANN_PACKAGE_PATH based on the ASCEND_HOME_PATH environment variable
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE STRING "ASCEND CANN package installation directory")

# Verify that the required paths exist
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist. Check whether the CANN package is installed in ${ASCEND_CANN_PACKAGE_PATH}")
endif()

# Include Ascend CANN CMake file
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Collect source files
file(GLOB ASCENDC_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

# Create an object library
ascendc_library(ascendc_kernels_npu STATIC ${ASCENDC_KERNEL_FILES})

set(ASCENDC_TARGET_NAME ascendc_kernels_npu PARENT_SCOPE)
set(ASCENDC_TARGET_DIR "${CMAKE_BINARY_DIR}/lib" PARENT_SCOPE)
