cmake_minimum_required(VERSION 3.11)

project(inferrt VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("-- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message("-- CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
endif()

set(CC gcc)
set(CXX g++)

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
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_SOURCE_DIR})

# Add compiler, ir, runtime and kernels directories
add_subdirectory(hardware)
add_subdirectory(ir)
add_subdirectory(lang)
add_subdirectory(ops)
add_subdirectory(optimize)
add_subdirectory(pybind)
add_subdirectory(runtime)

set(OBJECTS lexer_obj parser_obj ir_obj compiler_obj vm_obj runtime_obj ops_obj pass_obj mrt_ir_obj)

# Create da execution file
add_executable(da lang/cli/main.cc lang/cli/options.cc)
target_link_libraries(da ${OBJECTS} stdc++fs)

# Create shared library. target_compile_options(inferrt PRIVATE -fPIC)
add_library(inferrt SHARED lang/api/c_api.cc)
target_link_libraries(inferrt ${OBJECTS})
