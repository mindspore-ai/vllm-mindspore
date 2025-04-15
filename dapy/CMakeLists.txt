cmake_minimum_required(VERSION 3.10)
project(DaPy VERSION 1.0)
message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)

# Build Debug
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Include DALang headers
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dac)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dair)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dart)

# Include DaPy directories
include_directories(${PROJECT_SOURCE_DIR}/dalang_py)

# Include pybind11 directories
message("-- PYBIND11_PATH: ${PYBIND11_PATH}")
include_directories(${PYBIND11_PATH}/include)

# Find pybind11 package location, or call find_package(pybind11 REQUIRED)
add_subdirectory(${PYBIND11_PATH})
pybind11_add_module(_dapy NO_EXTRAS dalang_py/pybind11_api.cc)

message("-- DALANG_LIBRARIES: ${DALANG_LIBRARIES}")
target_link_libraries(_dapy PUBLIC ${DALANG_LIBRARIES})