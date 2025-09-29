if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH /usr/local/Ascend)
endif()

include_directories(${ASCEND_PATH}/latest/include/)
include_directories(${ASCEND_PATH}/latest/lib64/)
link_directories(${ASCEND_PATH}/latest/lib64/)

add_subdirectory(aclnn)
