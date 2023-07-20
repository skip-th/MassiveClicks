cmake_minimum_required(VERSION 3.18)
project(mclicks VERSION 1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -g0")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_62,code=compute_62 --generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_80,code=compute_80 -rdc=true")

set(CUDA_LIB_PATH /usr/local/cuda-12.1/lib64)
link_directories(${CUDA_LIB_PATH})

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(SRCS main.cpp data/dataset.cpp click_models/evaluation.cpp parallel_em/communicator.cpp utils/macros.cpp utils/timer.cpp)
set(CU_SRCS utils/utils.cu parallel_em/parallel_em.cu parallel_em/kernel.cu data/search.cu
            click_models/base.cu click_models/param.cu click_models/common.cu click_models/factor.cu
            click_models/pbm.cu click_models/ccm.cu click_models/dbn.cu click_models/ubm.cu)

add_executable(${PROJECT_NAME} ${SRCS} ${CU_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX CUDA::cudart pthread)
