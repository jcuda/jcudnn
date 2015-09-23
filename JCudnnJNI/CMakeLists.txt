cmake_minimum_required(VERSION 3.1)

project(JCudnn)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommon_CMake.txt)

if (NOT TARGET JCudaCommonJNI)
    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../jcuda-common/JCudaCommonJNI
        ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif() 

set (BUILD_SHARED_LIBS ON)

include_directories (
    src/
    ${JCudaCommonJNI_INCLUDE_DIRS}
    ${JNI_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
  
cuda_add_library(${PROJECT_NAME}
    src/JCudnn.cpp
)

include_directories (${CUDA_cuDNN_INCLUDE} REQUIRED)
find_library(CUDA_cuDNN_LIBRARY NAMES libcudnn.so cudnn.lib
             DOC "cuDNN library")
target_link_libraries(${PROJECT_NAME} ${CUDA_cuDNN_LIBRARY})

target_link_libraries(${PROJECT_NAME}
    JCudaCommonJNI
)


set_target_properties(${PROJECT_NAME} 
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-${CMAKE_HOST}-${CMAKE_ARCH})
