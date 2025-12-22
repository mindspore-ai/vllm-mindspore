check_debug_log_out()

#Collect all source files
file(GLOB OPS_LOWERED_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

#Create shared library for lowered kernel support
add_library(ops_ascend_lowered SHARED ${OPS_LOWERED_SRC_FILES})

# Find LLVM for SHA256 support
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})

#Link dependencies
target_link_libraries(ops_ascend_lowered PRIVATE
    hardware_ascend
    kernel
    mrt::securec
    ${CMAKE_DL_LIBS}
    LLVMSupport)

#Install library
install(TARGETS ops_ascend_lowered LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES
    lowered_op_helper.h
    DESTINATION include/ops/ascend/lowered)
