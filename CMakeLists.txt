cmake_minimum_required(VERSION 3.18)
project(powerpkg CUDA CXX)

# Detect CUDA compiler automatically
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA compiler and flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ compiler and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/problems)
include_directories(${CUDAToolkit_INCLUDE_DIRS})


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
    stdc++ 
    m     
)

target_include_directories(powerpkg
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)