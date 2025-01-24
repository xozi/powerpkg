cmake_minimum_required(VERSION 3.18)
project(powerpkg CUDA CXX)

# Set CUDA compiler and flags
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "-ccbin /usr/bin/gcc-13 --allow-unsupported-compiler")
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-13)
# GPU: 1650
set(CMAKE_CUDA_ARCHITECTURES 75)  

# Set C++ compiler and flags
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/problems)

file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/problems/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# Add your source files
add_executable(powerpkg src/main.cu)

target_link_libraries(powerpkg
    PRIVATE
    cublas
    cudart
    stdc++  # C++ standard library
    m       # Math library
)

target_include_directories(powerpkg
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)