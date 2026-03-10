#!/bin/bash
##################################################
# LLVM Build Script
# Required env vars: BUILD_JOBS, INC_BUILD
# Exports : LLVM_BUILD_DIR, LLVM_DIR, MLIR_DIR.
##################################################

_LLVM_ENV="${BUILD_DIR}/llvm_env.sh"

# Create temporary build directory for cmake
_TEMP_BUILD="${BUILD_DIR}/llvm_build_temp"

if [[ $INC_BUILD == 1 ]] && [[ -f "${_LLVM_ENV}" ]]; then
    echo "Reused persisted LLVM environment ${_LLVM_ENV}"
    source "${_LLVM_ENV}"
    return
fi

rm -rf "${_TEMP_BUILD}"
mkdir -p "${_TEMP_BUILD}"
# Create minimal CMakeLists.txt to trigger cmake/llvm.cmake
cat > "${_TEMP_BUILD}/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(llvm_builder NONE)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/llvm.cmake)
EOF

_CMAKE_ARGS=()
if [[ -n "${CMAKE_PREFIX_PATH}" ]]; then
    _CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
fi

cd "${_TEMP_BUILD}"
cmake . "${_CMAKE_ARGS[@]}"

# Source the generated environment file
if [[ -f "${_TEMP_BUILD}/set_env.sh" ]]; then
    echo "Saved LLVM environment variables to ${_LLVM_ENV}"
    cp "${_TEMP_BUILD}/set_env.sh" "${_LLVM_ENV}"
    source "${_LLVM_ENV}"
fi

# Clean up temporary directory (return to project root first)
cd "${PROJECT_DIR}"
rm -rf "${_TEMP_BUILD}"
