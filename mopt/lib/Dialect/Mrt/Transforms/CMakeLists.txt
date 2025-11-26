add_mlir_dialect_library(MLIRMrtTransforms
  SetTensorDevice.cc
  ConvertRankedTensorToMrtTensor.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt

  DEPENDS
  MLIRMrtTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRMrtDialect
)
