check_debug_log_out()

if(ENABLE_ASCEND)
    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH /usr/local/Ascend)
    endif()

    include_directories(${ASCEND_PATH}/latest/include/)
    include_directories(${ASCEND_PATH}/latest/lib64/)
    include_directories(${ASCEND_PATH}/latest/aarch64-linux/include/experiment)
    link_directories(${ASCEND_PATH}/latest/lib64/)

    if(ENABLE_TORCH_FRONT)
        add_compile_definitions(ENABLE_TORCH_NPU)
        link_directories(${TORCH_NPU_LIB_PATH})
        include_directories(${TORCH_NPU_INCLUDE})
    endif()
endif()

add_subdirectory(op_def)

add_library(kernel SHARED kernel_lib.cc op_register.cc custom_op_register.cc operator.cc)
target_link_libraries(kernel PRIVATE mrt_common mrt_ir_obj ${CMAKE_DL_LIBS})

set_target_properties(kernel PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS kernel
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

add_subdirectory(op_base)
add_subdirectory(cpu)

if(ENABLE_ASCEND)
    add_subdirectory(ascend)
endif()