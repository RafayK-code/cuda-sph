cmake_minimum_required(VERSION 3.18)
project(CudaSPH LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")

add_subdirectory(extern)

add_executable(cuda_sph
    src/main.cpp
    src/SPH.cu

    src/SPH.h
    src/Constants.h
    src/FluidMaths.cuh

    src/2D/SPH2D.h
    src/2D/SPH2D.cu
    src/2D/SPHCompute2D.cuh
    src/2D/SPHCompute2D.cu
    src/2D/SpatialHash2D.cuh
)

# CUDA-specific compile options
target_compile_options(cuda_sph PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --expt-relaxed-constexpr
        -Xcompiler=/Zc:preprocessor
    >
)

target_link_libraries(cuda_sph PRIVATE
    glfw
    glad
    imgui
    ${CUDA_LIBRARIES}
)

target_include_directories(cuda_sph PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui
    ${CUDAToolkit_INCLUDE_DIRS}
)