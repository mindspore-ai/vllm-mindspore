set(LLVM_LINK_COMPONENTS
  Core
  Support
  TransformUtils
)

add_llvm_executable(mrt-opt MrtOpt.cc)

llvm_update_compile_flags(mrt-opt)
target_link_libraries(mrt-opt PRIVATE
  MLIRAnalysis
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIROptLib
  MLIRTransforms
  MLIRFuncDialect
  MLIRMrtDialect
)

mlir_check_link_libraries(mrt-opt)

install(TARGETS mrt-opt
  EXPORT mrt-exports
  RUNTIME DESTINATION bin
)