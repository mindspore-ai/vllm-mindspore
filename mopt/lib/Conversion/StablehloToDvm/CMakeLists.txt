add_mlir_conversion_library(MoptStablehloToDvm
  StablehloToDvm.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/StablehloToDvm

  DEPENDS
  MoptStablehloToDvmIncGen
  MLIRDvmIncGen

  LINK_LIBS PUBLIC
  MLIRDvmDialect
  StablehloOps
  MLIRPass
  MLIRTransforms
)

