#!/bin/bash
##################################################
# LLVM Build Script
# Required env vars: BUILD_JOBS, INC_BUILD
# Exports : LLVM_SOURCE_DIR, LLVM_BUILD_DIR, LLVM_DIR, MLIR_DIR, TORCHMLIR_SOURCE_DIR, TORCHMLIR_BUILD_DIR, STABLEHLO_SOURCE_DIR.
##################################################

_LLVM_ENV="${BUILD_DIR}/llvm_env.sh"

# Create temporary build directory for cmake
_TEMP_BUILD="${BUILD_DIR}/llvm_build_temp"

if [[ $INC_BUILD == 1 ]] && [[ -f "${_LLVM_ENV}" ]]; then
    echo "Reused persisted LLVM environment ${_LLVM_ENV}"
    source "${_LLVM_ENV}"
    return
fi

mkdir -p "${_TEMP_BUILD}"
# Create minimal CMakeLists.txt to trigger cmake/llvm.cmake
cat > "${_TEMP_BUILD}/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(llvm_builder NONE)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/llvm.cmake)
EOF

cd "${_TEMP_BUILD}"
cmake .

# Source the generated environment file
if [[ -f "${_TEMP_BUILD}/set_env.sh" ]]; then
    echo "Saved LLVM environment variables to ${_LLVM_ENV}"
    cp "${_TEMP_BUILD}/set_env.sh" "${_LLVM_ENV}"
    source "${_LLVM_ENV}"
fi

# Clean up temporary directory (return to project root first)
cd "${PROJECT_DIR}"
rm -rf "${_TEMP_BUILD}"

if [[ $INC_BUILD != 1 ]]; then
    echo "Configuring LLVM (source: ${LLVM_SOURCE_DIR})"
    cmake -G Ninja -B "${LLVM_BUILD_DIR}" \
            ${CCACHE_CMAKE_ARGS} \
            -DCMAKE_BUILD_TYPE=Release \
            -DPython3_FIND_VIRTUALENV=ONLY \
            -DLLVM_ENABLE_PROJECTS=mlir \
            -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
            -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${TORCHMLIR_SOURCE_DIR}" \
            -DSTABLEHLO_EXTERNAL_DIR="${STABLEHLO_SOURCE_DIR}" \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DLLVM_TARGETS_TO_BUILD=host \
            -DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=ON \
            "${LLVM_SOURCE_DIR}/llvm"
fi
echo "Building LLVM and Torch-MLIR (${BUILD_JOBS} jobs)..."
cmake --build "${LLVM_BUILD_DIR}" -j "${BUILD_JOBS}"

