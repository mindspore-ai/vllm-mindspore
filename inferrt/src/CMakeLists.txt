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

# nanobind and Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
                OUTPUT_VARIABLE nanobind_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.include_dir())"
                OUTPUT_VARIABLE NANOBIND_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# Add compiler, ir, runtime and kernels directories
add_subdirectory(common)
add_subdirectory(config)
add_subdirectory(hardware)
add_subdirectory(ir)
add_subdirectory(ops)
add_subdirectory(optimize)
add_subdirectory(profiler)
add_subdirectory(runtime)

set(OBJECTS mrt_common mrt_config hardware_abstract execution_engine ops_obj pass_obj mrt_ir_obj ops_utils mrt_profiler)

# Create shared library. target_compile_options(inferrt PRIVATE -fPIC)
add_library(inferrt SHARED common/common.h)
set_target_properties(inferrt PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(inferrt PRIVATE ${NANOBIND_INCLUDE_DIR})
target_link_libraries(inferrt ${OBJECTS} nanobind-static Python::Python)

set_target_properties(inferrt PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS inferrt
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
