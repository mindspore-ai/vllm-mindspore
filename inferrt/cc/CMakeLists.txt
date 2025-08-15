cmake_minimum_required(VERSION 3.11)

project(inferrt VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("-- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Build Debug
if("${DEBUG}" STREQUAL "on")
set(CMAKE_BUILD_TYPE "Debug")
else()
set(CMAKE_BUILD_TYPE "Release")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
# Set PIC option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # add_compile_options(-fPIC)

# Include directories
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/c)
include_directories(${PROJECT_SOURCE_DIR}/ir)
include_directories(${PROJECT_SOURCE_DIR}/rt)
include_directories(${PROJECT_SOURCE_DIR}/api)

# Add compiler, ir, runtime and kernels directories
add_subdirectory(c)
add_subdirectory(ir)
add_subdirectory(kernels)
add_subdirectory(rt)

set(OBJECTS lexer_obj parser_obj ir_obj compiler_obj vm_obj tensor_obj runtime_obj ops_obj pass_obj stdc++fs dl)

# Create da execution file
add_executable(da cli/main.cc cli/options.cc)
target_link_libraries(da ${OBJECTS})

# Create shared library. target_compile_options(inferrt PRIVATE -fPIC)
add_library(inferrt SHARED api/c_api.cc)
target_link_libraries(inferrt ${OBJECTS})
