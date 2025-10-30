add_lit_testsuite(check-mrt
  "Running Mrt tests"
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS mrt-opt
  PARAMS
    llvm_tools_dir=${LLVM_BINARY_DIR}/bin
    mrt_tools_dir=${CMAKE_BINARY_DIR}/bin
    mrt_obj_root=${CMAKE_BINARY_DIR}
)
