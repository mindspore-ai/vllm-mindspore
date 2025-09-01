check_debug_log_out()

include(FetchContent)
if(ENABLE_GITEE)
  set(PKG_GIT_REPO https://github.com/google/googletest.git)
else()
  set(PKG_GIT_REPO https://gitee.com/mirrors/googletest.git)
endif()
FetchContent_Declare(
    googletest
    GIT_REPOSITORY ${PKG_GIT_REPO}
    GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)

message(STATUS "Build UT testcases")
include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)
file(GLOB_RECURSE STORAGE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../inferrt/src/ir/tensor/storage.cc"
        "../../../inferrt/src/hardware/hardware_abstract/*.cc"
        "../../../inferrt/src/hardware/cpu/*.cc"
        )
add_library(STORAGE_STATIC STATIC ${STORAGE_SRC_FILES})
target_link_libraries(STORAGE_STATIC mrt_common)

add_executable(run_storage_cpu_test ./ir/storage_cpu_test.cc)
target_link_libraries(run_storage_cpu_test
  PRIVATE
  -Wl,--whole-archive
  STORAGE_STATIC
  -Wl,--no-whole-archive
  GTest::gtest_main
  ${CMAKE_DL_LIBS}
)
add_test(NAME storage_cpu_test COMMAND run_storage_cpu_test)


if(ENABLE_ASCEND)
  message(STATUS "Backend Ascend enabled")
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_PATH /usr/local/Ascend)
  endif()

  message("Note compile ascend path: ${ASCEND_PATH}")
  include_directories(${ASCEND_PATH}/latest/include/)
  link_directories(${ASCEND_PATH}/latest/lib64/)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")

  include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)

  add_executable(run_hardware_ascend_test ./hardware/hardware_test.cc)
  target_link_libraries(run_hardware_ascend_test hardware_ascend)

  enable_testing()
  add_test(NAME hardware_ascend_test COMMAND run_hardware_ascend_test)
else()
  message("Note ascend backend is disabled")
endif()


if(ENABLE_CPU)
  message(STATUS "Backend CPU enabled")
  include_directories(${PROJECT_SOURCE_DIR}/inferrt/src)
  add_executable(run_hardware_cpu_test ./hardware/hardware_cpu_test.cc)
  target_link_libraries(run_hardware_cpu_test hardware_cpu)
  enable_testing()
  add_test(NAME hardware_cpu_test COMMAND run_hardware_cpu_test)
else()
  message("Note cpu backend is disabled")
endif()