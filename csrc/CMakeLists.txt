cmake_minimum_required(VERSION 3.18)
project(astrai_kernels LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TORCH_HOME)
    set(TORCH_HOME "$ENV{TORCH_HOME}")
endif()
if(NOT TORCH_HOME)
    message(FATAL_ERROR "TORCH_HOME must point at the torch install dir (site-packages/torch)")
endif()

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    set(PYTHON_INCLUDE_DIR "/usr/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()

if(NOT DEFINED ASTRAI_CUDA_ARCH)
    if(DEFINED ENV{ASTRAI_CUDA_ARCH})
        set(ASTRAI_CUDA_ARCH "$ENV{ASTRAI_CUDA_ARCH}")
    else()
        set(ASTRAI_CUDA_ARCH 80)
    endif()
endif()

set(TORCH_LIB_DIR "${TORCH_HOME}/lib")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64")

set(CXX_FLAGS -O3 -funroll-loops)
set(NVCC_FLAGS -O3
    --expt-relaxed-constexpr
    --use_fast_math
    "--ptxas-options=-O3,-v"
    --extra-device-vectorization
    --threads=16)

set(TORCH_LIBS
    "${TORCH_LIB_DIR}/libtorch_python.so"
    "${TORCH_LIB_DIR}/libtorch_cuda.so"
    "${TORCH_LIB_DIR}/libc10_cuda.so"
    "${TORCH_LIB_DIR}/libtorch_cpu.so"
    "${TORCH_LIB_DIR}/libtorch.so"
    "${TORCH_LIB_DIR}/libc10.so"
    CUDA::cudart)

set(CMAKE_CUDA_ARCHITECTURES "${ASTRAI_CUDA_ARCH}")

set(KERNELS attn_decode attn_prefill attn_paged_decode attn_paged_prefill rotary_emb fp8_mm)

foreach(name ${KERNELS})
    add_library(${name} MODULE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${name}.cu")

    target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name})

    target_include_directories(${name} PRIVATE
        "${TORCH_HOME}/include"
        "${TORCH_HOME}/include/torch/csrc/api/include"
        "${PYTHON_INCLUDE_DIR}")

    target_link_libraries(${name} PRIVATE ${TORCH_LIBS})
    target_link_options(${name} PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")

    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)

    set_target_properties(${name} PROPERTIES
        PREFIX ""
        SUFFIX ".${PY_SOABI}.so"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../astrai/extension/lib")
endforeach()
