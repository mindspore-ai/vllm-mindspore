cmake_minimum_required(VERSION 3.10)
project(dart VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CC gcc)
set(CXX g++)

add_subdirectory(${PROJECT_SOURCE_DIR}/dalang)
add_subdirectory(${PROJECT_SOURCE_DIR}/dapy)