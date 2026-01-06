check_debug_log_out()
find_package(Threads REQUIRED)
execute_process(COMMAND python -c "import pybind11; print(pybind11.__path__[0])"
                OUTPUT_VARIABLE PYBIND11_INSTALL_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYBIND11_INSTALL_PATH}")
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB_RECURSE RUNTIME_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(runtime_obj STATIC ${RUNTIME_SRC_FILES})
target_link_libraries(runtime_obj ops_obj mrt_ir_obj kernel Threads::Threads hardware_abstract Python3::Python)
target_include_directories(runtime_obj PRIVATE ${pybind11_INCLUDE_DIRS})
