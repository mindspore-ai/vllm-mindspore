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
set(TORCHMLIR_VERSION "2024.05.0" CACHE INTERNAL "Torch-MLIR version")
set(TORCHMLIR_COMMIT "7e7af670802d99cacdaf26e6e37249d544e4896e" CACHE INTERNAL "Torch-MLIR commit hash")
set(TORCHMLIR_SHA256 "a8ff6ae22031349ecf5c9ab089091ab12861cec736a795689682b28a79faaaf8")
set(TORCHMLIR_URL "https://gitee.com/dayschan/torch-mlir/repository/archive/${TORCHMLIR_COMMIT}.zip")

# Download Torch-MLIR source and apply patches (SOURCEMODULES only downloads, no build)
mrt_add_pkg(torch_mlir
    VER ${TORCHMLIR_VERSION}
    URL ${TORCHMLIR_URL}
    SHA256 ${TORCHMLIR_SHA256}
    SOURCEMODULES .
    CUSTOM_SUBMODULE_INFO ${TORCHMLIR_COMMIT}
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/001-build-isolate-symbols.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/002-build-external-stablehlo.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/003-build-remove-tests.patch
    PATCHES ${TOP_DIR}/third_party/patch/torch-mlir/004-build-embedded.patch
)

message(STATUS "Torch-MLIR source downloaded: ${torch_mlir_DIRPATH}")

