
file(GLOB_RECURSE COMMON_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(mrt_common SHARED ${COMMON_SRC_FILES})
target_link_libraries(mrt_common dl mrt::glog)

set_target_properties(mrt_common PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS mrt_common
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES ${glog_LIBPATH}/libglog.so.0.7.1
        DESTINATION lib
        RENAME libglog.so.2)
