# Generate pattern rewrite rules from TableGen
set(LLVM_TARGET_DEFINITIONS StablehloToMrtPatterns.td)
mlir_tablegen(StablehloToMrtPatterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(MoptStablehloToMrtPatternsIncGen)

add_mlir_conversion_library(MoptConversionStablehloToMrt
  StablehloToMrt.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/StablehloToMrt

  DEPENDS
  MoptConversionPassIncGen
  MoptStablehloToMrtPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  MLIRMrtDialect
  StablehloOps
)

# Add generated include directory
target_include_directories(MoptConversionStablehloToMrt PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

