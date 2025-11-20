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

# Some upstream packages only populate *_CMAKE_DIR when building in-tree.
# Fall back to the config directories we just located to make sure module lookups work.
if(NOT MLIR_CMAKE_DIR)
  set(MLIR_CMAKE_DIR "${MLIR_DIR}")
endif()
if(NOT LLVM_CMAKE_DIR)
  set(LLVM_CMAKE_DIR "${LLVM_DIR}")
endif()

# Set CMAKE_MODULE_PATH early, as StableHLO also needs these modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(TOP_DIR "${PROJECT_SOURCE_DIR}/..")

# ===========================================================================
# Torch-MLIR Integration (build alongside mopt)
# ===========================================================================
# Note: StableHLO will be compiled by Torch-MLIR, not separately by mopt.
# This avoids duplicate compilation and ensures consistent stablehlo versions.

include(${TOP_DIR}/cmake/stablehlo.cmake)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(TORCHMLIR_SOURCE_DIR "${TORCHMLIR_SOURCE_DIR}" CACHE PATH "Path to Torch-MLIR sources")
set(STABLEHLO_SOURCE_DIR "${STABLEHLO_SOURCE_DIR}" CACHE PATH "Path to StableHLO sources")

message(STATUS "Using Torch-MLIR source: ${TORCHMLIR_SOURCE_DIR}")
message(STATUS "Using StableHLO source: ${STABLEHLO_SOURCE_DIR}")

# Enable position-independent code for embedded torch-mlir build
# This is required when static libraries are linked into shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure Torch-MLIR to compile StableHLO
# StableHLO will be compiled by Torch-MLIR and available through its Python extensions
set(TORCH_MLIR_BUILD_EMBEDDED ON CACHE BOOL "Build Torch-MLIR as embedded" FORCE)
set(TORCH_MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable Python bindings" FORCE)
set(TORCH_MLIR_ENABLE_STABLEHLO ON CACHE BOOL "Enable StableHLO support (compiled by Torch-MLIR)" FORCE)
set(USE_STABLEHLO_SOURCE_DIR "${STABLEHLO_SOURCE_DIR}" CACHE PATH "Path to external StableHLO sources" FORCE)
set(TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS OFF CACHE BOOL "Disable PyTorch extensions" FORCE)
set(TORCH_MLIR_ENABLE_REFBACKEND OFF CACHE BOOL "Disable RefBackend" FORCE)
set(TORCH_MLIR_ENABLE_LTC OFF CACHE BOOL "Disable LTC" FORCE)
set(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER OFF CACHE BOOL "Disable JIT IR Importer" FORCE)

add_subdirectory(${TORCHMLIR_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/torch-mlir EXCLUDE_FROM_ALL)

include_directories("${TORCHMLIR_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/torch-mlir/include")
# StableHLO headers (compiled by Torch-MLIR)
include_directories("${STABLEHLO_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/torch-mlir/stablehlo")

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
