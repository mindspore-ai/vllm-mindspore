# ===========================================================================
# Torch-MLIR Source Download Module
# ===========================================================================
# Downloads Torch-MLIR source and applies patches using mrt_add_pkg
# ===========================================================================

# Load mrt_add_pkg
if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

# ===========================================================================
# Download Torch-MLIR
# ===========================================================================
message(STATUS "Downloading Torch-MLIR...")

# Torch-MLIR version info
set(TORCHMLIR_VERSION "2025.11.18" CACHE INTERNAL "Torch-MLIR daily version")
set(TORCHMLIR_COMMIT "b834f94badeefcb046f60a2bda51b6c3591cb21b" CACHE INTERNAL "Torch-MLIR commit hash")
set(TORCHMLIR_SHA256 "531e6841ad86a32eeb6da84165847379c290f34c4820823a821ad4c5f8efc1e4")
set(TORCHMLIR_URL "https://gitee.com/dayschan/torch-mlir/repository/archive/${TORCHMLIR_COMMIT}.zip")

# Download Torch-MLIR source and apply patches (SOURCEMODULES only downloads, no build)
mrt_add_pkg(torch_mlir
    VER ${TORCHMLIR_VERSION}
    URL ${TORCHMLIR_URL}
    SHA256 ${TORCHMLIR_SHA256}
    SOURCEMODULES .
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/001-build-isolate-symbols.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/002-build-external-stablehlo.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/003-build-remove-tests.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/004-build-embedded.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/005-skip-operator-op-check.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/006-support-floordiv-ceildiv-symint.patch
)

message(STATUS "Torch-MLIR source downloaded: ${torch_mlir_DIRPATH}")
