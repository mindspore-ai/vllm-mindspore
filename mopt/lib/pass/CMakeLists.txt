add_mlir_library(MoptPasses
  ReshapeConversionPass.cc

  DEPENDS
  MoptPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRIR
  MLIRTransformUtils
  MLIRRewrite
  MLIRArithDialect
  MLIRMrtDialect
)

# Add include directories
target_include_directories(MoptPasses PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
