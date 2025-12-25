add_mlir_conversion_library(MoptStablehloToDvm
  StablehloToDvm.cc
  ConvertDvmToMrtDvmCall.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/StablehloToDvm

  DEPENDS
  MoptStablehloToDvmIncGen
  MoptConversionPassIncGen
  MLIRDvmIncGen

  LINK_LIBS PUBLIC
  MLIRDvmDialect
  MLIRMrtDialect
  MLIRFuncDialect
  StablehloOps
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

