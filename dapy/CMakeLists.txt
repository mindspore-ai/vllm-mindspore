cmake_minimum_required(VERSION 3.11)
project(DaPy VERSION 1.0)
message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Include DALang headers
include_directories(${PROJECT_SOURCE_DIR}/../dalang)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dac)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dair)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/dart)
include_directories(${PROJECT_SOURCE_DIR}/../dalang/api)

# Include DaPy directories
include_directories(${PROJECT_SOURCE_DIR}/dalang_py)

# Prepare pybind11 module
set(depname "pybind11")
set(PYBIND11_PATH "${PROJECT_SOURCE_DIR}/${depname}-src")
message("-- PYBIND11_PATH: ${PYBIND11_PATH}")

# Download and copy pybind11 project if not exists
if(NOT EXISTS ${PYBIND11_PATH})
    message("-- Downloading ${depname} module...")
    include(FetchContent)
    FetchContent_Declare(
        ${depname}
        # Change github repo to gitee's: https://github.com/pybind/pybind11 ==> https://gitee.com/mirrors/pybind11
        GIT_REPOSITORY https://gitee.com/mirrors/pybind11.git
        GIT_TAG 58c382a8e3d7081364d2f5c62e7f429f0412743b # stable
    )
    FetchContent_MakeAvailable(${depname})
    message("-- pybind11_SOURCE_DIR: ${${depname}_SOURCE_DIR}")
    message("-- pybind11_BINARY_DIR: ${${depname}_BINARY_DIR}")
    # Find pybind11 package location, or call find_package(pybind11 REQUIRED)
    message("-- Copying ${${depname}_SOURCE_DIR} to ${PROJECT_SOURCE_DIR}/...")
    file(COPY ${${depname}_SOURCE_DIR} DESTINATION ${PROJECT_SOURCE_DIR})
endif()

# Include pybind11 directories
include_directories(${PYBIND11_PATH}/include)

# Add pybind11 sub module
add_subdirectory(${PYBIND11_PATH})
pybind11_add_module(_dapy NO_EXTRAS dalang_py/pybind11_api.cc)
target_link_libraries(_dapy PUBLIC dalang)