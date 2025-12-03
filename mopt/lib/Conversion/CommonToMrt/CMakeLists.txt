file(GLOB COMMON_TO_MRT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MoptConversionCommonToMrt
  ${COMMON_TO_MRT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/CommonToMrt

  DEPENDS
  MoptConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRMrtDialect
)
