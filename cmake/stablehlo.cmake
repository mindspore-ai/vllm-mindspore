# ===========================================================================
# StableHLO Source Download Module
# ===========================================================================
# Downloads StableHLO source code via mrt_add_pkg so it can be reused by both
# torch-mlir (as an external project) and mopt (for in-tree builds).
# ===========================================================================

if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

message(STATUS "Downloading StableHLO sources...")

set(STABLEHLO_VERSION "2025.10.23" CACHE INTERNAL "StableHLO daily version")
set(STABLEHLO_COMMIT "4c0d4841519aed22e3689c30b72a0e4228051249" CACHE INTERNAL "StableHLO commit hash")
set(STABLEHLO_SHA256 "b49cfd0987d5a2a82d007059503e70209b68e049a30231d2e8b0b9389e51d6e3")
set(STABLEHLO_URL "https://gitee.com/dayschan/stablehlo/repository/archive/${STABLEHLO_COMMIT}.zip")

mrt_add_pkg(stablehlo
    VER ${STABLEHLO_VERSION}
    URL ${STABLEHLO_URL}
    SHA256 ${STABLEHLO_SHA256}
    SOURCEMODULES .
    PATCHES ${TOP_DIR}/third_party/patch/stablehlo/001-fix-compiler-warning-flags.patch
    PATCHES ${TOP_DIR}/third_party/patch/stablehlo/002-fix-compile-error-with-old-gcc.patch
)

set(STABLEHLO_SOURCE_DIR ${stablehlo_DIRPATH})

message(STATUS "StableHLO source downloaded: ${STABLEHLO_SOURCE_DIR}")
