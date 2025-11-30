file(GLOB STABLEHLO_TO_MRT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MoptConversionStablehloToMrt
  ${STABLEHLO_TO_MRT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/StablehloToMrt

  DEPENDS
  MoptConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRMrtDialect
  StablehloOps
)
