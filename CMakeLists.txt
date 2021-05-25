cmake_minimum_required(VERSION 3.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/../out")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O3 -Wall")


find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(OpenCV_DIR /usr/local/opencv3.4/share/OpenCV)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})


include_directories(${PROJECT_SOURCE_DIR}/common)
aux_source_directory(${PROJECT_SOURCE_DIR}/common COMMON_SRC)


# add_subdirectory(RGBToGray)
# add_subdirectory(DeviceInfo)
add_subdirectory(basic_compute)