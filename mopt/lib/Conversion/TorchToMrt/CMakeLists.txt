# Generate pattern rewrite rules from TableGen
# set(LLVM_TARGET_DEFINITIONS TorchToMrtPatterns.td)
# mlir_tablegen(TorchToMrtPatterns.cpp.inc -gen-rewriters)
# add_public_tablegen_target(MoptTorchToMrtPatternsIncGen)

add_mlir_conversion_library(MoptConversionTorchToMrt
  TorchToMrt.cc

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/TorchToMrt

  DEPENDS
  MoptConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  MLIRMrtDialect
  TorchMLIRTorchDialect
)

# Add generated include directory
target_include_directories(MoptConversionTorchToMrt PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
