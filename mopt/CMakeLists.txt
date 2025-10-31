check_debug_log_out()

# Find packages (requires installed *Config.cmake files)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "Found MLIR at ${MLIR_DIR}")

# Common LLVM compile definitions
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# MLIR CMake helpers
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)          # If you need ODS/TableGen
include(AddLLVM)           # LLVM helper macros
include(AddMLIR)           # MLIR helper macros

file(GLOB_RECURSE PASS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(mopt SHARED ${PASS_SRC_FILES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Choose MLIR/LLVM components to link
# Available components can be inspected in install/lib/cmake/mlir/*.cmake
target_link_libraries(mopt
  PRIVATE
    MLIRIR
    MLIRParser
    MLIROptLib
    MLIRSupport
    MLIRTransforms
    MLIRPass
    MLIRDialectUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRTosaDialect
    MLIRRewrite
    LLVMCore
    LLVMSupport
)

# Include directories (MLIR installed headers)
target_include_directories(mopt
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_options(mopt PRIVATE -Wl,-rpath,$ORIGIN/../_vendor/llvm/lib)

mlir_check_all_link_libraries(mopt)
