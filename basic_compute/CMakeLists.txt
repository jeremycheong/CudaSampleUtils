project(BasicCompute)

set(CULibName "basic_compute")
file(GLOB CU_HEADS ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB CU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
cuda_add_library(${CULibName} SHARED ${CU_HEADS} ${CU_SOURCE})
target_link_libraries(${CULibName} ${OpenCV_LIBS})


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC)

add_executable(${PROJECT_NAME} ${SRC} ${COMMON_SRC})
target_link_libraries(${PROJECT_NAME} ${CULibName})