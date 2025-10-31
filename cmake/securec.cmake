set(securec_USE_STATIC_LIBS ON)

if(MSVC)
    # add "/EHsc", for vs2019 warning C4530 about securec
    set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS}")
endif()

# libboundscheck-v1.1.16
set(REQ_URL "https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.zip")
set(SHA256 "5119bda1ee96440c1a45e23f0cb8b079cc6697e052c4a78f27d0869f84ba312b")

mrt_add_pkg(securec
        VER 1.1.16
        LIBS securec
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION} -DTARGET_OHOS_LITE=OFF
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/securec/securec.patch001
        )

include_directories(${securec_INC})
include_directories(${securec_INC}/../)
add_library(mrt::securec ALIAS securec::securec)
