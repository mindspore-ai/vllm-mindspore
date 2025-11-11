check_debug_log_out()

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

if(ENABLE_TORCH_FRONT)
    add_subdirectory(cpu/aten)
endif()

if(ENABLE_ASCEND)
    add_subdirectory(ascend)
endif()