check_debug_log_out()

#Collect all source files
file(GLOB OPS_LOWERED_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

#Create shared library for lowered kernel support
add_library(ops_ascend_lowered SHARED ${OPS_LOWERED_SRC_FILES})

#Link dependencies
target_link_libraries(ops_ascend_lowered PRIVATE
    ops_base_obj hardware_ascend kernel mrt::securec ${CMAKE_DL_LIBS} crypto)

#Install library
install(TARGETS ops_ascend_lowered LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES
    kernel_spec.h lowered_kernel_executor.h auto_lowered_op.h
    mlir_compiler.h lowered_op_helper.h
    DESTINATION include/ops/ascend/lowered)
