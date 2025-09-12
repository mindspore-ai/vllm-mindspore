cmake_minimum_required(VERSION 3.11)
project(inferrt VERSION 1.0)

message("-- CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}")
message("-- CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  message("-- CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
endif()

set(CC gcc)
set(CXX g++)

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()
message("-- CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

function(check_debug_log_out)
  get_filename_component(CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if("${DEBUG_LOG_OUT_${CURRENT_FOLDER}}" STREQUAL "on")
    message("-- Enable log out for source in ${CMAKE_CURRENT_SOURCE_DIR}")
    add_definitions(-DDEBUG_LOG_OUT)
  endif()
endfunction()

if(ENABLE_TORCH_FRONT)
  execute_process(COMMAND python -c "import torch; print(torch.compiled_with_cxx11_abi())" OUTPUT_VARIABLE PYTORCH_CXX11_ABI_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

  if("${PYTORCH_CXX11_ABI_VERSION}" STREQUAL "True")
    message("-- Enable _GLIBCXX_USE_CXX11_ABI")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  else()
    message("-- Disable _GLIBCXX_USE_CXX11_ABI")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  endif()
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/inferrt/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)