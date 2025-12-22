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
