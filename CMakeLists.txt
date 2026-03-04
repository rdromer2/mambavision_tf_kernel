cmake_minimum_required(VERSION 3.14)
project(MambaVision_TF_Kernel)

# Asegurar que se usará C++17 como requiere TF en configuraciones modernas
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ACTIVACIÓN DEL ECOSISTEMA CUDA
# Se le dice al orquestador que busque el ejecutable "nvcc" en el entorno (ej. Colab).
enable_language(CUDA)

# Encontrar Python y el paquete de TensorFlow
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Ejecutar un script para obtener los flags de TF (Include y Link)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TF_INC
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Añadir directorio de inclusión de headers de Tensorflow
include_directories(${TF_INC})

# ==============================================================================
# LIBRERÍA 1: FORWARD PASS
# ==============================================================================
add_library(mamba_kernel SHARED mamba_ssm_op.cc mamba_ssm_kernel.cu)

target_compile_options(mamba_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC -O2>
)

set_target_properties(mamba_kernel PROPERTIES CUDA_ARCHITECTURES "75;80")

separate_arguments(TF_COMPILE_FLAGS_LIST NATIVE_COMMAND "${TF_COMPILE_FLAGS_STR}")
separate_arguments(TF_LINK_FLAGS_LIST NATIVE_COMMAND "${TF_LINK_FLAGS_STR}")

target_compile_options(mamba_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TF_COMPILE_FLAGS_LIST}>
)
target_link_options(mamba_kernel PRIVATE ${TF_LINK_FLAGS_LIST})

# ==============================================================================
# LIBRERÍA 2: BACKWARD PASS (FASE 4)
# ==============================================================================
# Empaquetamos el nuevo wrapper y kernel en una librería dinámica independiente
# para no alterar el binario original del forward pass ya verificado.
add_library(mamba_bwd_kernel SHARED mamba_ssm_bwd_op.cc mamba_ssm_bwd_kernel.cu)

target_compile_options(mamba_bwd_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC -O2>
)

# Misma arquitectura de hardware que el forward pass (T4 y A100)
set_target_properties(mamba_bwd_kernel PROPERTIES CUDA_ARCHITECTURES "75;80")

# Inyectamos los mismos flags nativos de TF
target_compile_options(mamba_bwd_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TF_COMPILE_FLAGS_LIST}>
)
target_link_options(mamba_bwd_kernel PRIVATE ${TF_LINK_FLAGS_LIST})
