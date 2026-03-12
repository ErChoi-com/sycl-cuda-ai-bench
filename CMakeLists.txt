cmake_minimum_required(VERSION 3.23)
project(sycl_cuda_ai_bench LANGUAGES CXX)

include(CheckLanguage)

option(BUILD_CUDA "Build CUDA backend" ON)
option(BUILD_SYCL "Build SYCL backend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(benchmark_common
    src/common/benchmark_common.cpp
)

target_include_directories(benchmark_common PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(benchmark_cpu
    src/benchmark_main.cpp
    src/backends/cpu_backend.cpp
)
target_compile_definitions(benchmark_cpu PRIVATE ENABLE_CPU_BACKEND)
target_link_libraries(benchmark_cpu PRIVATE benchmark_common)
target_compile_options(benchmark_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)

if(BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_executable(benchmark_cuda
            src/benchmark_main.cpp
            src/backends/cuda_backend.cu
        )
        target_compile_definitions(benchmark_cuda PRIVATE ENABLE_CUDA_BACKEND)
        target_link_libraries(benchmark_cuda PRIVATE benchmark_common)
        set_target_properties(benchmark_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        target_compile_options(benchmark_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
        )
    else()
        message(WARNING "CUDA requested but no CUDA toolset was found. Skipping benchmark_cuda target.")
    endif()
endif()

if(BUILD_SYCL)
    find_program(DPCPP_COMPILER NAMES dpcpp icpx)
    if(DPCPP_COMPILER)
        add_executable(benchmark_sycl
            src/benchmark_main.cpp
            src/backends/sycl_backend.cpp
        )
        target_compile_definitions(benchmark_sycl PRIVATE ENABLE_SYCL_BACKEND)
        target_link_libraries(benchmark_sycl PRIVATE benchmark_common)

        # Keep compile flags target-local so this CMake file works with non-oneAPI host compilers too.
        target_compile_options(benchmark_sycl PRIVATE -fsycl -O3)
        target_link_options(benchmark_sycl PRIVATE -fsycl)
    else()
        message(WARNING "SYCL requested but dpcpp/icpx was not found. Skipping benchmark_sycl target.")
    endif()
endif()
