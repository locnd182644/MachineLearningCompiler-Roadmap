cmake_minimum_required(VERSION 3.16)
project(week1_roofline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(memcpy_bench memcpy_bench.cpp)

# Bật tập lệnh của CPU đang biên dịch (AVX2/FMA...) nếu trình biên dịch hỗ trợ.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(memcpy_bench PRIVATE -march=native)
endif()
