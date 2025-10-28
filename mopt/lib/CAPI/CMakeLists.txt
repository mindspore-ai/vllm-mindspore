add_mlir_public_c_api_library(MLIRMrtCAPI
  Registration.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRMrtDialect
)

mopt_target_includes(MLIRMrtCAPI)