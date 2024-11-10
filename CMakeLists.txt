cmake_minimum_required(VERSION 3.10)

project(DaLang VERSION 1.0)

message("-- CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Build Debug
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(lexer)
add_subdirectory(parser)

# Test for dalang
add_executable(dalang test/main.cc)
target_link_libraries(dalang lexer_obj parser_obj)