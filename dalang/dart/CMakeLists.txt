if("${DEBUG_LOG_OUT_rt}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

add_subdirectory(runtime)