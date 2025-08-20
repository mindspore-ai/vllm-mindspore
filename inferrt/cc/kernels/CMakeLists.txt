add_library(kernels SHARED kernel_lib.cc)
target_link_libraries(kernels PRIVATE tensor_obj dl)

add_subdirectory(dummy)

if(ENABLE_KERNELS_ATEN)
    add_subdirectory(aten)
endif()
