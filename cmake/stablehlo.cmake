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

set(STABLEHLO_VERSION "0.17.0" CACHE INTERNAL "StableHLO version")
set(STABLEHLO_COMMIT "c28d55e91b4a5daaff18a33ce7e9bbd0f171256a" CACHE INTERNAL "StableHLO commit hash")
set(STABLEHLO_SHA256 "26270c26da86f97ec8f011e6c35c021a932fa1cde7ccc110dd9ae29f73dc814a")
set(STABLEHLO_URL "https://gitee.com/dayschan/stablehlo/repository/archive/${STABLEHLO_COMMIT}.zip")

mrt_add_pkg(stablehlo
    VER ${STABLEHLO_VERSION}
    URL ${STABLEHLO_URL}
    SHA256 ${STABLEHLO_SHA256}
    SOURCEMODULES .
    CUSTOM_SUBMODULE_INFO ${STABLEHLO_COMMIT}
)

message(STATUS "StableHLO source downloaded: ${STABLEHLO_SOURCE_DIR}")
