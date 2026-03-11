cmake_minimum_required(VERSION 3.11)
project(DaPy VERSION 1.0)
message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

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
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

check_debug_log_out()

# Link against nanobind libraries
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add ms_inferrt pybind11 sub module
add_subdirectory(ms_inferrt)

# Add ms_inferrt_torch pybind11 sub module
if(ENABLE_TORCH_FRONT)
    add_subdirectory(ms_inferrt_torch)
endif()
