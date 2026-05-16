cmake_minimum_required(VERSION 3.18)
project(week6_memory_tiling CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(matmul_3level matmul_tiled_3level.cu)
  set_target_properties(matmul_3level PROPERTIES CUDA_ARCHITECTURES "native")
  # Khi so sánh với cuBLAS, bỏ comment:
  # find_package(CUDAToolkit REQUIRED)
  # target_link_libraries(matmul_3level PRIVATE CUDA::cublas)
else()
  message(STATUS "Không tìm thấy CUDA toolkit — bỏ qua matmul_tiled_3level.cu")
endif()
