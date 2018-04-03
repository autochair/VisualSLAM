CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(VisionWorks REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
#find_package(VisionWorks-NVXIO REQUIRED)

include_directories("${VisionWorks_INCLUDE_DIRS}")
#include_directories("${VisionWorks-NVXIO_INCLUDE_DIRS}")
include_directories(include)
include_directories(/home/nvidia/VisionWorks-1.6-Samples/nvxio/include)


add_executable( test_demo src/main_hough_transform.cpp
  )
target_link_libraries(test_demo
  /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libnvx.a
  /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libovx.a
  /usr/local/cuda/targets/aarch64-linux/lib/libcudart.so
  ${CUDA_LIBRARIES}
  ${CUDA_LIBRARY}
  ${VisionWorks_LIBRARIES}

  gstpbutils-1.0
  gstapp-1.0 
  gstbase-1.0 
  gstreamer-1.0 
  gobject-2.0 
  glib-2.0  

  GLU
  GL
  glut
    )
