file(GLOB MRT_TRANSFORMS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_dialect_library(MLIRMrtTransforms
  ${MRT_TRANSFORMS_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt

  DEPENDS
  MLIRMrtTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRMrtDialect
)
