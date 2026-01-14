# Fusion passes for StableHLO to MRT Linalg conversion
add_mlir_library(MoptFusionPasses
  OutlineStablehloFusionRegions.cpp
  MarkTorchToStablehloOp.cpp
  SymbolicShape.cpp
  PropagateTorchSymbolicShape.cpp
  RemoveRedundantDynamicBroadcast.cpp
  VerifySymbolicShapeAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt/Fusion

  DEPENDS
  MoptFusionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRShapeDialect
  MLIRSupport
  MLIRTransformUtils
  StablehloOps
  TorchMLIRTorchDialect
  TorchMLIRTorchConversionDialect
)

# Add generated include directory
target_include_directories(MoptFusionPasses PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
