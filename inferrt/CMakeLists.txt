cmake_minimum_required(VERSION 3.11)
project(inferrt VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)


function(check_debug_log_out)
    get_filename_component(CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    if("${DEBUG_LOG_OUT_${CURRENT_FOLDER}}" STREQUAL "on")
        message("-- Enable log out for source in ${CMAKE_CURRENT_SOURCE_DIR}")
        add_definitions(-DDEBUG_LOG_OUT)
    endif()
endfunction()


add_subdirectory(${PROJECT_SOURCE_DIR}/cc)
add_subdirectory(${PROJECT_SOURCE_DIR}/py)