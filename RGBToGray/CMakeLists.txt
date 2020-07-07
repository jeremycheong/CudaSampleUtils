cmake_minimum_required(VERSION 3.5)

project(RGBToGray)

find_package(CUDA REQUIRED)

set(OpneCV_DIR /usr/local/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

set(CULibName "rgbgray")
file(GLOB CU_HEADS ${CMAKE_SOURCE_DIR}/*.cuh)
file(GLOB CU_SOURCE ${CMAKE_SOURCE_DIR}/*.cu)
cuda_add_library(${CULibName} SHARED ${CU_HEADS} ${CU_SOURCE})
target_link_libraries(${CULibName} ${OpenCV_LIBS})

aux_source_directory(${CMAKE_SOURCE_DIR} SRC)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} ${CULibName})