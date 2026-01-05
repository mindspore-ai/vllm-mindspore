
add_mlir_pdll_library(TorchToMrtPDLLIncGen
  TorchToMrt.pdll
  TorchToMrt.pdll.h.inc

  DEPENDS
  TorchToMrtAten.pdll
  TorchToMrtConstant.pdll
  TorchToMrtNpu.pdll
)

file(GLOB TORCH_TO_MRT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_mlir_conversion_library(MoptConversionTorchToMrt
  ${TORCH_TO_MRT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/include/mopt/Conversion/TorchToMrt

  DEPENDS
  MoptConversionPassIncGen
  TorchToMrtPDLLIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  MLIRMrtDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRParser
)

# Add generated include directory
target_include_directories(MoptConversionTorchToMrt PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
