# DVM operator module CMakeLists.txt
#
# The DVM library is now managed by cmake/dvm.cmake and automatically downloaded
# from https://repo.mindspore.cn/mindspore/dvm/ during configuration.

# DVM source files
file(GLOB DVM_SRCS *.cc)

# Create DVM operator library.
# Keep it consistent with other Ascend op modules: build as a shared library
# that links against `kernel` and can be deployed/loaded at runtime.
add_library(ops_ascend_dvm SHARED ${DVM_SRCS})

# Include DVM headers
target_include_directories(ops_ascend_dvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    # For includes like "ops/op_register.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Link with pre-built DVM library (managed by cmake/dvm.cmake)
target_link_libraries(ops_ascend_dvm PRIVATE mrt::dvm)

# Link with other dependencies
target_link_libraries(ops_ascend_dvm PRIVATE
    hardware_ascend
    kernel
    mrt::securec
)

# Add compile options
target_compile_options(ops_ascend_dvm PRIVATE
    -Wall
    -Werror
)

set_target_properties(ops_ascend_dvm PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_dvm
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

message(STATUS "DVM operator module configured successfully")
