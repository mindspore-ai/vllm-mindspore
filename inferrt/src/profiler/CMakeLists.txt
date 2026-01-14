# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set the source files for profiler
set(PROFILER_SOURCES
  profiler.cc
)

# Create object library
add_library(mrt_profiler SHARED ${PROFILER_SOURCES})

# Include directories
target_include_directories(mrt_profiler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../ir
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
)

# Link with dependencies
target_link_libraries(mrt_profiler PRIVATE mrt::json)
set_target_properties(mrt_profiler PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(TARGETS mrt_profiler
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
