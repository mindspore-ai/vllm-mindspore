cmake_minimum_required(VERSION 3.10)

project(DaPy VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Build Debug
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -shared -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -shared -O3 -Wall")
# Set PIC option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # add_compile_options(-fPIC)


# Include Python headers directories
message("-- CMAKE_PYTHON_INCLUDE_DIR: ${CMAKE_PYTHON_INCLUDE_DIR}")
message("-- CMAKE_PYTHON_CONFIG_CFLAGS: ${CMAKE_PYTHON_CONFIG_CFLAGS}")
message("-- CMAKE_PYTHON_CONFIG_LDFLAGS: ${CMAKE_PYTHON_CONFIG_LDFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_PYTHON_INCLUDE_DIR} ${CMAKE_PYTHON_CONFIG_LDFLAGS}")

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
message("-- PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("-- PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})


# Include DALang headers
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dac)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dair)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dart)

# Include DaPy directories
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/dalang)

message("-- CMAKE_DALANG_LIB_PATH: ${CMAKE_DALANG_LIB_PATH}")
add_executable(dapy dalang/main.cc dalang/pybind11_api.cc)
target_link_libraries(dapy ${CMAKE_DALANG_LIB_PATH} ${PYTHON_LIBRARIES})
target_compile_options(dapy PRIVATE -fPIC)
