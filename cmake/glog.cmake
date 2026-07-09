if(NOT COMMAND mrt_add_pkg)
    include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)
endif()

if(MSVC)
    set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Dgoogle=mrt_private /EHsc")
else()
    set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Dgoogle=mrt_private")
endif()
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(DEFINED PYTORCH_CXX11_ABI_VERSION)
    if("${PYTORCH_CXX11_ABI_VERSION}" STREQUAL "True")
        set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    else()
        set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.7.1.tar.gz")
set(SHA256 "54854d52a4a0f12a7a57f43d22457477281ef373b6487c5ac422e6303d7ff3e8")

set(glog_option
    -DBUILD_TESTING=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=ON
    -DWITH_GFLAGS=OFF
    -DCMAKE_BUILD_TYPE=Release
)

mrt_add_pkg(glog
    VER 0.7.1
    LIBS glog
    URL ${REQ_URL}
    SHA256 ${SHA256}
    CMAKE_OPTION ${glog_option}
)

include_directories(${glog_INC})
add_library(mrt::glog ALIAS glog::glog)
add_compile_definitions(USE_GLOG)
