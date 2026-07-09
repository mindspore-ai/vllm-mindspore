# DVM operator module CMakeLists.txt
#
# The DVM library is now managed by cmake/dvm.cmake and automatically downloaded
# from https://repo.mindspore.cn/mindspore/dvm/ during configuration.

# DVM source files
file(GLOB DVM_SRCS *.cc)
set(DVM_V2_SRCS op_dvm_call_v2.cc op_dvm_call_v2_register.cc)
foreach(_v2_src ${DVM_V2_SRCS})
    list(REMOVE_ITEM DVM_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${_v2_src}")
endforeach()

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
    mrt_common
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

string(TOUPPER "${ENABLE_DVM_V2}" _enable_dvm_v2)
if(NOT _enable_dvm_v2 STREQUAL "AUTO" AND NOT _enable_dvm_v2 STREQUAL "ON" AND NOT _enable_dvm_v2 STREQUAL "OFF")
    message(FATAL_ERROR "ENABLE_DVM_V2 must be AUTO, ON, or OFF, but got: ${ENABLE_DVM_V2}")
endif()

set(_build_dvm_v2 FALSE)
set(_dvm_v2_disable_reason "")
if(_enable_dvm_v2 STREQUAL "OFF")
    set(_dvm_v2_disable_reason "ENABLE_DVM_V2=OFF")
else()
    mrt_check_dvm_v2_package(_dvm_v2_package_available _dvm_v2_package_reason)
    set(DVM_TORCH_NPU_PYBIND_API "${TORCH_NPU_INCLUDE}/torch_npu/csrc/inductor/dvm/pybind_api.h")
    if(NOT _dvm_v2_package_available)
        set(_dvm_v2_disable_reason "${_dvm_v2_package_reason}")
    elseif(NOT TORCH_NPU_INCLUDE)
        set(_dvm_v2_disable_reason "TORCH_NPU_INCLUDE is not configured")
    elseif(NOT EXISTS "${DVM_TORCH_NPU_PYBIND_API}")
        set(_dvm_v2_disable_reason "torch-npu DVM pybind header was not found at ${DVM_TORCH_NPU_PYBIND_API}")
    else()
        set(_build_dvm_v2 TRUE)
    endif()
endif()

if(NOT _build_dvm_v2)
    if(_enable_dvm_v2 STREQUAL "ON")
        message(FATAL_ERROR "DVM V2 is required by ENABLE_DVM_V2=ON, but it is unavailable: "
                            "${_dvm_v2_disable_reason}")
    endif()
    message(STATUS "DVM V2 operator module disabled: ${_dvm_v2_disable_reason}")
else()
    add_library(ops_ascend_dvm_v2 SHARED ${DVM_V2_SRCS})

    mrt_create_dvm_torch_npu_include_shim(DVM_TORCH_NPU_INCLUDE_SHIM)

    target_include_directories(ops_ascend_dvm_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${DVM_TORCH_NPU_INCLUDE_SHIM}
        ${DVM_V2_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${TORCH_NPU_INCLUDE}
    )

    target_link_libraries(ops_ascend_dvm_v2 PRIVATE
        mrt::dvm_v2
        ops_ascend_aclnn_common
        ops_utils
        torch_npu
        hardware_ascend
        kernel
        mrt_ir_obj
        mrt::securec
        Python::Python
    )

    target_compile_options(ops_ascend_dvm_v2 PRIVATE
        -Wall
        -Werror
        -Wno-error=unused-function
        -Wno-error=unused-variable
    )

    set_target_properties(ops_ascend_dvm_v2 PROPERTIES
      INSTALL_RPATH "$ORIGIN;${TORCH_NPU_LIB_PATH};${TORCH_INSTALL_PREFIX}/lib"
      BUILD_WITH_INSTALL_RPATH TRUE
    )

    install(TARGETS ops_ascend_dvm_v2
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

message(STATUS "DVM operator module configured successfully")
