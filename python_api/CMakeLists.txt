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

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")

# Set PIC option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # add_compile_options(-fPIC)

# Include Python directories
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
include_directories(${PROJECT_SOURCE_DIR}/dalang_py)

message("-- DALANG_LIBRARIES: ${DALANG_LIBRARIES}")
add_executable(dapy dalang_py/pybind11_api.cc)
target_link_libraries(dapy ${DALANG_LIBRARIES} ${PYTHON_LIBRARIES})
target_compile_options(dapy PRIVATE -fPIC)
