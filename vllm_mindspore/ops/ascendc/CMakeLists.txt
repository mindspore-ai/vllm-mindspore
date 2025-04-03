cmake_minimum_required(VERSION 3.16)
project(AscendC_Kernels)

# Parameters passed from command line or default values
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
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

# Add source files
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/adv_step_flash.c)

# Build shared library
ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

# Set the output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)