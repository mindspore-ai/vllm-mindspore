
# Include MLIR Python CMake utilities
include(AddMLIRPython)
find_package(pybind11 CONFIG REQUIRED)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mopt.")

# Define main Python package sources
declare_mlir_python_sources(MoptPythonSources)

# Define dialect Python bindings
declare_mlir_dialect_python_bindings(
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mopt"
  ADD_TO_PARENT MoptPythonSources
  TD_FILE dialects/MrtBinding.td
  SOURCES dialects/mrt/__init__.py
  DIALECT_NAME mrt
)

# Define Python extension module
declare_mlir_python_extension(MoptPythonExtension
  MODULE_NAME _mopt
  SOURCES MoptModule.cc
  EMBED_CAPI_LINK_LIBS
    MLIRMrtCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

set(_source_components
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MoptPythonSources
    MoptPythonExtension
)

add_mlir_python_common_capi_library(MoptAggregateCAPI
  INSTALL_COMPONENT MoptPythonPackage
  INSTALL_DESTINATION python_packages/mopt/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/mopt/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

# Create Python package
add_mlir_python_modules(MoptPythonPackage
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/mopt"
  INSTALL_PREFIX "python_packages/mopt"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    MoptAggregateCAPI
)