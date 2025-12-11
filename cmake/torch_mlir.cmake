# ===========================================================================
# Torch-MLIR out-of-tree build module
# ===========================================================================
# Downloads Torch-MLIR, builds it out-of-tree against pre-built LLVM/MLIR,
# and prepares a Python wheel for reuse.
# ===========================================================================

if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

message(STATUS "Configuring Torch-MLIR (out-of-tree)...")

set(TORCHMLIR_VERSION "2025.11.18" CACHE INTERNAL "Torch-MLIR daily version")
set(TORCHMLIR_COMMIT "b834f94badeefcb046f60a2bda51b6c3591cb21b" CACHE INTERNAL "Torch-MLIR commit hash")
set(TORCHMLIR_SHA256 "531e6841ad86a32eeb6da84165847379c290f34c4820823a821ad4c5f8efc1e4")
set(TORCHMLIR_URL "https://gitee.com/dayschan/torch-mlir/repository/archive/${TORCHMLIR_COMMIT}.zip")

set(_TORCHMLIR_CMAKE_OPTIONS
    -DCMAKE_BUILD_TYPE=Release
    -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON
    -DPython3_FIND_VIRTUALENV=ONLY
    -DPython_FIND_VIRTUALENV=ONLY
    -DTORCH_MLIR_ENABLE_STABLEHLO=OFF
    -DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=ON
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=torch_mlir
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DLLVM_DIR=${LLVM_DIR}
    -DMLIR_DIR=${MLIR_DIR}
)

mrt_add_pkg(torch_mlir
    VER ${TORCHMLIR_VERSION}
    URL ${TORCHMLIR_URL}
    SHA256 ${TORCHMLIR_SHA256}
    CMAKE_PATH .
    CMAKE_OPTION ${_TORCHMLIR_CMAKE_OPTIONS}
    CUSTOM_CMAKE_GENERATOR Ninja
    SOURCEMODULES .
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/001-build-isolate-symbols.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/002-build-external-stablehlo.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/003-build-remove-tests.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/004-build-embedded.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/005-skip-operator-op-check.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/006-support-floordiv-ceildiv-symint.patch
)

set(TORCHMLIR_SOURCE_DIR ${torch_mlir_DIRPATH})
set(TORCHMLIR_INSTALL_DIR "${torch_mlir_DIRPATH}/_install")

message(STATUS "Torch-MLIR source directory: ${TORCHMLIR_SOURCE_DIR}")
message(STATUS "Torch-MLIR install directory: ${TORCHMLIR_INSTALL_DIR}")

# Write torch_mlir install path to file for setup.py
file(WRITE ${CMAKE_BINARY_DIR}/torch_mlir_install_path.txt "${TORCHMLIR_INSTALL_DIR}")
