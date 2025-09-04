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

if(TARGET pybind11::headers)
    message("-- pybind11 already included")
else()
    # Prepare pybind11 module
    message("-- pybind11 not included, start including")
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

    add_subdirectory(${PYBIND11_PATH})
endif()

# Add mrt pybind11 sub module
add_subdirectory(mrt)

# Add mrt_torch pybind11 sub module
if(ENABLE_KERNEL_ATEN)
    add_subdirectory(mrt_torch)
endif()
