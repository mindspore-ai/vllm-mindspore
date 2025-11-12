cmake_minimum_required(VERSION 3.20.0)
project(mrt-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Set CMake policies to avoid warnings
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)  # Use new FindPython module instead of FindPythonInterp
endif()
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 OLD)  # Maintain old add_custom_command behavior for StableHLO compatibility
endif()

function(mopt_target_includes target)
  set(_dirs
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )
  target_include_directories(${target} PUBLIC ${_dirs})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} PRIVATE ${_dirs})
  endif()
endfunction()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found at: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set CMAKE_MODULE_PATH early, as StableHLO also needs these modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# ===========================================================================
# StableHLO Integration - Using independent cmake configuration file
# ===========================================================================
include(${PROJECT_SOURCE_DIR}/../cmake/stablehlo.cmake)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# CMAKE_MODULE_PATH already set earlier (required by StableHLO)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include/mopt)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
