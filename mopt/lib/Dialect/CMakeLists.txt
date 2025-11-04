add_mlir_dialect_library(MLIRMrtDialect
  MrtDialect.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt

  DEPENDS
  MLIRMrtIncGen
  MLIRMrtExtraIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)