check_debug_log_out()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_FOUND: ${Python3_FOUND}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
include_directories(${Python3_INCLUDE_DIRS})

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

endif()

file(GLOB_RECURSE HARDWARE_ABSTRACT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(hardware_abstract_obj STATIC ${HARDWARE_ABSTRACT_SRC_FILES})