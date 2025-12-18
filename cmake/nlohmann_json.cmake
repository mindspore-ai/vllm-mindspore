if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

message(STATUS "Configuring nlohmann_json library...")

set(NLOHMANN_JSON_VERSION "3.10.1" CACHE INTERNAL "nlohmann_json version")

set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.10.1.zip")
set(SHA256 "5c7d0a0542431fef628f8dc4c34fd022fe8747ccb577012d58f38672d8747e0d")

# Check cache directory
set(NLOHMANN_JSON_CACHE_DIR ${_MRT_LIB_CACHE}/nlohmann_json3101_${NLOHMANN_JSON_VERSION})

if(EXISTS ${NLOHMANN_JSON_CACHE_DIR}/options.txt)
    message(STATUS "nlohmann_json library found in cache: ${NLOHMANN_JSON_CACHE_DIR}")
endif()

mrt_add_pkg(nlohmann_json3101
    VER 3.10.1
    HEAD_ONLY ./include
    URL ${REQ_URL}
    SHA256 ${SHA256}
)

include_directories(${nlohmann_json3101_INC})
add_library(mrt::json ALIAS nlohmann_json3101)
