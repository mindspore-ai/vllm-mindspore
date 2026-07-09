# ===========================================================================
# DVM Pre-built Library
# ===========================================================================
# Downloads pre-built DVM static libraries (.a files) for Ascend NPU operations
# Note: dvm.h header file is kept in src/ops/ascend/dvm/prebuild/ directory
# ===========================================================================

if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

message(STATUS "Configuring DVM library...")

set(DVM_VERSION "r2.7_20251127" CACHE INTERNAL "DVM version")
set(DVM_COMMIT "1941ab79f9c3641b9e7a9c049b2b6a4e75de92b6" CACHE INTERNAL "DVM commit hash")
string(CONCAT DVM_BASE_URL
    "https://repo.mindspore.cn/mindspore/dvm/daily/202511/20251127/"
    "r2.7_20251127183657_${DVM_COMMIT}/ascend"
)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(DVM_ARCH "x86_64")
    set(DVM_URL "${DVM_BASE_URL}/x86_64/libdvm.a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(DVM_ARCH "aarch64")
    set(DVM_URL "${DVM_BASE_URL}/aarch64/libdvm.a")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Only x86_64 and aarch64 are supported.")
endif()

message(STATUS "DVM architecture: ${DVM_ARCH}")

# dvm.h header is kept in the source tree (stable interface)
set(DVM_INCLUDE_DIR ${TOP_DIR}/inferrt/src/ops/ascend/dvm/prebuild)


# Download from remote
set(DVM_CACHE_DIR ${_MRT_LIB_CACHE}/dvm_${DVM_VERSION}_${DVM_ARCH})

if(NOT EXISTS ${DVM_CACHE_DIR}/libdvm.a)
    message(STATUS "Downloading DVM library from: ${DVM_URL}")
    file(MAKE_DIRECTORY ${DVM_CACHE_DIR})

    # Download libdvm.a
    file(DOWNLOAD
        ${DVM_URL}
        ${DVM_CACHE_DIR}/libdvm.a
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )

    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if(NOT DOWNLOAD_RESULT EQUAL 0)
        list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
        message(FATAL_ERROR "Failed to download DVM library: ${DOWNLOAD_ERROR}")
    endif()

    # Write version info
    file(WRITE ${DVM_CACHE_DIR}/lib_info.txt
        "[lib information]\ngit branch: r2.7\ncommit  id: ${DVM_COMMIT}\n")

    message(STATUS "DVM library downloaded successfully")
else()
    message(STATUS "DVM library found in cache: ${DVM_CACHE_DIR}")
endif()

set(DVM_LIBRARY ${DVM_CACHE_DIR}/libdvm.a)

# Verify library exists
if(NOT EXISTS ${DVM_LIBRARY})
    message(FATAL_ERROR "DVM library not found at: ${DVM_LIBRARY}")
endif()

# Verify header exists
if(NOT EXISTS ${DVM_INCLUDE_DIR}/dvm.h)
    message(FATAL_ERROR "DVM header not found at: ${DVM_INCLUDE_DIR}/dvm.h")
endif()

message(STATUS "DVM library: ${DVM_LIBRARY}")
message(STATUS "DVM include directory: ${DVM_INCLUDE_DIR}")

# Create imported target
add_library(mrt::dvm STATIC IMPORTED GLOBAL)
set_target_properties(mrt::dvm PROPERTIES
    IMPORTED_LOCATION ${DVM_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${DVM_INCLUDE_DIR}
)

set(DVM_V2_VERSION "r2.10_20260124" CACHE STRING "DVM v2 version used by dvm_call_v2")
set(DVM_V2_COMMIT "34be3e58c8c6f9b3a559c40b62f80bdf2d2c1bfe" CACHE INTERNAL "DVM v2 commit hash")
set(DVM_V2_HEADER_COMMIT "7608eafc16b3f70ff327bd8a42ac99a365f71405" CACHE INTERNAL "DVM v2 header commit hash")
string(CONCAT DVM_V2_BASE_URL
    "https://repo.mindspore.cn/mindspore/dvm/daily/202601/20260124/"
    "master_20260124150902_${DVM_V2_COMMIT}/ascend"
)
set(DVM_V2_DEFAULT_URL "${DVM_V2_BASE_URL}/${DVM_ARCH}/libdvm.a")
set(DVM_V2_CACHE_DIR ${_MRT_LIB_CACHE}/dvm_${DVM_V2_VERSION}_${DVM_ARCH})
set(DVM_V2_HEADER_CACHE_DIR ${_MRT_LIB_CACHE}/dvm_headers_${DVM_V2_HEADER_COMMIT})
set(DVM_V2_HEADER_GIT_REPOSITORY "https://gitcode.com/mindspore/dvm.git" CACHE STRING "DVM v2 header git repository")
set(DVM_V2_URL "${DVM_V2_DEFAULT_URL}" CACHE STRING "URL to download the DVM v2 static library used by dvm_call_v2")
set(DVM_V2_ROOT "" CACHE PATH "Optional external DVM v2 package root used by dvm_call_v2")
set(DVM_V2_INCLUDE_DIR "" CACHE PATH "External DVM v2 include directory used by dvm_call_v2")
set(DVM_V2_USE_EXTERNAL_ROOT OFF CACHE BOOL "Use DVM_V2_ROOT instead of InferRT's managed DVM v2 dependency")
if(DEFINED ENV{DVM_V2_ROOT} AND NOT "$ENV{DVM_V2_ROOT}" STREQUAL "")
    set(DVM_V2_ROOT "$ENV{DVM_V2_ROOT}" CACHE PATH "Optional external DVM package root used by dvm_call_v2" FORCE)
    set(DVM_V2_USE_EXTERNAL_ROOT ON CACHE BOOL "Use DVM_V2_ROOT instead of InferRT's managed DVM v2 dependency" FORCE)
endif()
if(DEFINED ENV{DVM_V2_INCLUDE_DIR} AND NOT "$ENV{DVM_V2_INCLUDE_DIR}" STREQUAL "")
    set(DVM_V2_INCLUDE_DIR "$ENV{DVM_V2_INCLUDE_DIR}"
        CACHE PATH "External DVM v2 include directory used by dvm_call_v2" FORCE)
endif()
if(DEFINED ENV{DVM_V2_URL} AND NOT "$ENV{DVM_V2_URL}" STREQUAL "")
    set(DVM_V2_URL "$ENV{DVM_V2_URL}"
        CACHE STRING "URL to download the DVM v2 static library used by dvm_call_v2" FORCE)
elseif("${DVM_V2_URL}" STREQUAL "")
    set(DVM_V2_URL "${DVM_V2_DEFAULT_URL}"
        CACHE STRING "URL to download the DVM v2 static library used by dvm_call_v2" FORCE)
endif()

function(mrt_download_dvm_v2_headers out_dir out_reason)
    set(_downloaded_dir "")
    set(_reason "")
    if(EXISTS "${DVM_V2_HEADER_CACHE_DIR}/include/dvm.h" AND EXISTS "${DVM_V2_HEADER_CACHE_DIR}/include/dvm_py.h")
        set(_downloaded_dir "${DVM_V2_HEADER_CACHE_DIR}/include")
    else()
        find_program(_git_executable git)
        if(NOT _git_executable)
            set(_reason "git was not found, cannot fetch DVM v2 headers")
        else()
            file(REMOVE_RECURSE "${DVM_V2_HEADER_CACHE_DIR}")
            get_filename_component(_dvm_v2_header_cache_parent "${DVM_V2_HEADER_CACHE_DIR}" DIRECTORY)
            file(MAKE_DIRECTORY "${_dvm_v2_header_cache_parent}")
            message(STATUS "Downloading DVM v2 headers from: ${DVM_V2_HEADER_GIT_REPOSITORY}")
            execute_process(
                COMMAND "${_git_executable}" clone "${DVM_V2_HEADER_GIT_REPOSITORY}" "${DVM_V2_HEADER_CACHE_DIR}"
                RESULT_VARIABLE _clone_result
                OUTPUT_QUIET
                ERROR_VARIABLE _clone_error
            )
            if(_clone_result EQUAL 0)
                execute_process(
                    COMMAND "${_git_executable}" checkout --detach "${DVM_V2_HEADER_COMMIT}"
                    WORKING_DIRECTORY "${DVM_V2_HEADER_CACHE_DIR}"
                    RESULT_VARIABLE _checkout_result
                    OUTPUT_QUIET
                    ERROR_VARIABLE _checkout_error
                )
                if(_checkout_result EQUAL 0 AND EXISTS "${DVM_V2_HEADER_CACHE_DIR}/include/dvm.h"
                    AND EXISTS "${DVM_V2_HEADER_CACHE_DIR}/include/dvm_py.h")
                    set(_downloaded_dir "${DVM_V2_HEADER_CACHE_DIR}/include")
                    message(STATUS "DVM v2 headers downloaded successfully")
                else()
                    file(REMOVE_RECURSE "${DVM_V2_HEADER_CACHE_DIR}")
                    set(_reason "failed to checkout DVM v2 header commit ${DVM_V2_HEADER_COMMIT}: ${_checkout_error}")
                endif()
            else()
                file(REMOVE_RECURSE "${DVM_V2_HEADER_CACHE_DIR}")
                set(_reason "failed to clone DVM v2 header repository: ${_clone_error}")
            endif()
        endif()
    endif()
    set(${out_dir} "${_downloaded_dir}" PARENT_SCOPE)
    set(${out_reason} "${_reason}" PARENT_SCOPE)
endfunction()

function(mrt_find_dvm_v2_include_dir out_dir out_reason)
    set(_candidate_dirs)
    set(_checked_dirs)
    if(NOT "${DVM_V2_INCLUDE_DIR}" STREQUAL "" AND NOT "${DVM_V2_INCLUDE_DIR}" MATCHES "/prebuild_v2/?$")
        list(APPEND _candidate_dirs "${DVM_V2_INCLUDE_DIR}")
    endif()
    if(NOT "${DVM_V2_ROOT}" STREQUAL "")
        list(APPEND _candidate_dirs "${DVM_V2_ROOT}/include")
    endif()
    list(APPEND _candidate_dirs
        "${TOP_DIR}/../dvm/include"
        "${TOP_DIR}/../torch_npu/pytorch/third_party/dvm/dvm/include"
        "${TOP_DIR}/third_party/dvm/include"
        "${TOP_DIR}/third_party/dvm/dvm/include"
        "${TOP_DIR}/dvm/include"
    )

    set(_found_dir "")
    foreach(_dir ${_candidate_dirs})
        list(APPEND _checked_dirs "${_dir}")
        if(EXISTS "${_dir}/dvm.h" AND EXISTS "${_dir}/dvm_py.h")
            set(_found_dir "${_dir}")
            break()
        endif()
    endforeach()
    if("${_found_dir}" STREQUAL "")
        mrt_download_dvm_v2_headers(_downloaded_dir _download_reason)
        if(NOT "${_downloaded_dir}" STREQUAL "")
            set(_found_dir "${_downloaded_dir}")
        else()
            string(REPLACE ";" ", " _checked_dirs_msg "${_checked_dirs}")
            set(_reason "DVM v2 headers dvm.h and dvm_py.h were not found. Checked: ${_checked_dirs_msg}")
            if(NOT "${_download_reason}" STREQUAL "")
                set(_reason "${_reason}. ${_download_reason}")
            endif()
        endif()
    endif()
    set(${out_dir} "${_found_dir}" PARENT_SCOPE)
    set(${out_reason} "${_reason}" PARENT_SCOPE)
endfunction()

if(DVM_V2_USE_EXTERNAL_ROOT)
    if("${DVM_V2_ROOT}" STREQUAL "")
        message(FATAL_ERROR "DVM_V2_USE_EXTERNAL_ROOT=ON requires DVM_V2_ROOT to be set.")
    endif()
    set(DVM_V2_LIBRARY "${DVM_V2_ROOT}/libdvm.a" CACHE FILEPATH "DVM static library used by dvm_call_v2" FORCE)
    set(DVM_V2_INCLUDE_DIR "${DVM_V2_ROOT}/include" CACHE PATH "DVM include directory used by dvm_call_v2" FORCE)
else()
    if(NOT EXISTS "${DVM_V2_CACHE_DIR}/libdvm.a" AND NOT "${DVM_V2_URL}" STREQUAL "")
        message(STATUS "Downloading DVM v2 library from: ${DVM_V2_URL}")
        file(MAKE_DIRECTORY ${DVM_V2_CACHE_DIR})
        file(DOWNLOAD
            ${DVM_V2_URL}
            ${DVM_V2_CACHE_DIR}/libdvm.a
            SHOW_PROGRESS
            STATUS DVM_V2_DOWNLOAD_STATUS
        )
        list(GET DVM_V2_DOWNLOAD_STATUS 0 DVM_V2_DOWNLOAD_RESULT)
        if(NOT DVM_V2_DOWNLOAD_RESULT EQUAL 0)
            list(GET DVM_V2_DOWNLOAD_STATUS 1 DVM_V2_DOWNLOAD_ERROR)
            file(REMOVE ${DVM_V2_CACHE_DIR}/libdvm.a)
            message(WARNING "Failed to download DVM v2 library: ${DVM_V2_DOWNLOAD_ERROR}")
        else()
            file(WRITE ${DVM_V2_CACHE_DIR}/lib_info.txt
                "[lib information]\ngit branch: ${DVM_V2_VERSION}\nurl: ${DVM_V2_URL}\n")
            message(STATUS "DVM v2 library downloaded successfully")
        endif()
    elseif(EXISTS "${DVM_V2_CACHE_DIR}/libdvm.a")
        message(STATUS "DVM v2 library found in cache: ${DVM_V2_CACHE_DIR}")
    endif()
    set(DVM_V2_LIBRARY "${DVM_V2_CACHE_DIR}/libdvm.a" CACHE FILEPATH "DVM static library used by dvm_call_v2" FORCE)
    mrt_find_dvm_v2_include_dir(_dvm_v2_found_include_dir _dvm_v2_find_include_reason)
    set(DVM_V2_INCLUDE_DIR "${_dvm_v2_found_include_dir}"
        CACHE PATH "External DVM v2 include directory used by dvm_call_v2" FORCE)
    set(DVM_V2_INCLUDE_REASON "${_dvm_v2_find_include_reason}"
        CACHE INTERNAL "DVM v2 include directory resolution failure reason")
endif()
set(ENABLE_DVM_V2 "AUTO" CACHE STRING "Enable dvm_call_v2 support: AUTO, ON, or OFF")
set_property(CACHE ENABLE_DVM_V2 PROPERTY STRINGS AUTO ON OFF)

function(mrt_check_dvm_v2_package out_available out_reason)
    set(_available TRUE)
    set(_reason "")
    if(NOT EXISTS "${DVM_V2_LIBRARY}")
        set(_available FALSE)
        set(_reason "libdvm.a was not found at ${DVM_V2_LIBRARY}")
    elseif("${DVM_V2_INCLUDE_DIR}" STREQUAL "")
        set(_available FALSE)
        set(_reason "${DVM_V2_INCLUDE_REASON}")
    elseif(NOT EXISTS "${DVM_V2_INCLUDE_DIR}/dvm.h")
        set(_available FALSE)
        set(_reason "dvm.h was not found at ${DVM_V2_INCLUDE_DIR}/dvm.h")
    elseif(NOT EXISTS "${DVM_V2_INCLUDE_DIR}/dvm_py.h")
        set(_available FALSE)
        set(_reason "dvm_py.h was not found at ${DVM_V2_INCLUDE_DIR}/dvm_py.h")
    endif()
    set(${out_available} ${_available} PARENT_SCOPE)
    set(${out_reason} "${_reason}" PARENT_SCOPE)
endfunction()

function(mrt_require_dvm_v2)
    mrt_check_dvm_v2_package(_dvm_v2_available _dvm_v2_reason)
    if(NOT _dvm_v2_available)
        message(FATAL_ERROR "dvm_call_v2 requires the DVM package used by torch-npu's dvm.kernel layer, "
                            "but it is unavailable: ${_dvm_v2_reason}. "
                            "Check InferRT's managed DVM dependency, set DVM_V2_URL to override "
                            "the download URL, or set DVM_V2_ROOT and DVM_V2_USE_EXTERNAL_ROOT=ON "
                            "to use an external DVM package.")
    endif()

    message(STATUS "DVM v2 library: ${DVM_V2_LIBRARY}")
    message(STATUS "DVM v2 include directory: ${DVM_V2_INCLUDE_DIR}")

    if(NOT TARGET mrt::dvm_v2)
        add_library(mrt::dvm_v2 STATIC IMPORTED GLOBAL)
        set_target_properties(mrt::dvm_v2 PROPERTIES
            IMPORTED_LOCATION ${DVM_V2_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${DVM_V2_INCLUDE_DIR}
        )
    endif()
endfunction()

function(mrt_create_dvm_torch_npu_include_shim out_dir)
    mrt_require_dvm_v2()

    set(_shim_root "${CMAKE_BINARY_DIR}/generated/dvm_torch_npu_include")
    set(_shim_dir "${_shim_root}/third_party/dvm/dvm/include")
    file(MAKE_DIRECTORY "${_shim_dir}")
    file(WRITE "${_shim_dir}/dvm.h"
"#ifndef MRT_GENERATED_DVM_TORCH_NPU_DVM_H_\n"
"#define MRT_GENERATED_DVM_TORCH_NPU_DVM_H_\n"
"#include_next \"dvm.h\"\n"
"#endif  // MRT_GENERATED_DVM_TORCH_NPU_DVM_H_\n")
    file(WRITE "${_shim_dir}/dvm_py.h"
"#ifndef MRT_GENERATED_DVM_TORCH_NPU_DVM_PY_H_\n"
"#define MRT_GENERATED_DVM_TORCH_NPU_DVM_PY_H_\n"
"#include_next \"dvm_py.h\"\n"
"#endif  // MRT_GENERATED_DVM_TORCH_NPU_DVM_PY_H_\n")
    set(${out_dir} "${_shim_root}" PARENT_SCOPE)
endfunction()
