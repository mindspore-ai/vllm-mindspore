check_debug_log_out()

add_subdirectory(op_def)

add_library(kernel SHARED kernel_lib.cc op_register.cc)
target_link_libraries(kernel PRIVATE mrt_common mrt_ir_obj ${CMAKE_DL_LIBS})

add_subdirectory(dummy)
add_subdirectory(op_base)

if(ENABLE_TORCH_FRONT)
    add_subdirectory(cpu/aten)
endif()

if(ENABLE_ASCEND)
    add_subdirectory(ascend)
endif()
