# ===========================================================================
# StableHLO Integration Configuration - Independent Build Mode
# ===========================================================================
# Build StableHLO from source, completely isolated from torch_mlir
# This ensures torch_mlir and mopt each have their own StableHLO instance,
# avoiding global state conflicts
# ===========================================================================

message(STATUS "Configuring StableHLO integration (independent build)...")

# StableHLO source path (reuse source downloaded by torch-mlir)
set(STABLEHLO_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../third_party/torch-mlir/externals/stablehlo")

# Verify source exists
if(NOT EXISTS "${STABLEHLO_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "StableHLO source not found at: ${STABLEHLO_SOURCE_DIR}\n"
        "Please ensure torch-mlir submodule is initialized with StableHLO sources.")
endif()

message(STATUS "Using StableHLO source: ${STABLEHLO_SOURCE_DIR}")

# StableHLO build options
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "Build StableHLO as embedded" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable Python bindings" FORCE)
set(STABLEHLO_ENABLE_STRICT_BUILD OFF CACHE BOOL "Disable strict build" FORCE)

# Add StableHLO subdirectory (compile from source)
# Build in mopt's own build directory, completely independent of torch_mlir
add_subdirectory(${STABLEHLO_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/stablehlo EXCLUDE_FROM_ALL)

# Verify critical targets are available
if(NOT TARGET StablehloOps)
    message(FATAL_ERROR "StablehloOps target not found. StableHLO compilation failed.")
endif()
if(NOT TARGET ChloOps)
    message(FATAL_ERROR "ChloOps target not found. StableHLO compilation failed.")
endif()

# Add StableHLO header paths
include_directories("${STABLEHLO_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/stablehlo")

message(STATUS "StableHLO will be built independently (not shared with torch_mlir)")
message(STATUS "StableHLO configured successfully")
