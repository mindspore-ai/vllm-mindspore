add_subdirectory(op_def)

add_library(kernel SHARED kernel_lib.cc)
target_link_libraries(kernel PRIVATE mrt_ir_obj ${CMAKE_DL_LIBS})

add_subdirectory(dummy)

if(ENABLE_KERNEL_ATEN)
    add_subdirectory(cpu/aten)
endif()
