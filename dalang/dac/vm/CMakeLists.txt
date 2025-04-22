if("${DEBUG_LOG_OUT_vm}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

file(GLOB_RECURSE VM_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(vm_obj STATIC ${VM_SRC_FILES})
target_link_libraries(vm_obj compiler_obj)