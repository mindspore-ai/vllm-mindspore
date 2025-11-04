file(GLOB_RECURSE ATEN_KERNEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# Link against torch libraries.
execute_process(COMMAND python -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PATH}/share/cmake")
find_package(Torch REQUIRED)

add_library(kernel_aten SHARED ${ATEN_KERNEL_SRC_FILES})
target_link_libraries(kernel_aten PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(kernel_aten PRIVATE ops_obj ops_base_obj kernel)

set_target_properties(kernel_aten PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS kernel_aten
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)