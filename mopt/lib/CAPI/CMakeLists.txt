add_mlir_public_c_api_library(MoptCAPI
  Dialects.cc
  Passes.cc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mopt-c/

  DEPENDS
  MoptConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRMrtDialect
  MoptConversionStablehloToMrt
)

# Add include directories
target_include_directories(MoptCAPI PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

