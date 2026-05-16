cmake_minimum_required(VERSION 3.18)
project(week2_cpu_gpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- CPU: AVX matmul ---
add_executable(avx_matmul avx_matmul.cpp)
target_compile_options(avx_matmul PRIVATE -O3 -mavx2 -mfma)

# --- GPU: CUDA matmul (chỉ build nếu có CUDA toolkit) ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(matmul matmul.cu)
  set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "native")
  # Khi làm V3 cuBLAS, bỏ comment 2 dòng dưới:
  # find_package(CUDAToolkit REQUIRED)
  # target_link_libraries(matmul PRIVATE CUDA::cublas)
else()
  message(STATUS "Không tìm thấy CUDA toolkit — bỏ qua matmul.cu")
endif()
