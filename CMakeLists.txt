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

# Crear la biblioteca compartida empaquetando el controlador en C++ y el Kernel en .cu.
# CMake delegará inteligentemente cada archivo `.cu` a *nvcc* y cada `.cc` a *g++*,
# y luego linkeará sus resultados en un objeto dinámico (.so) unificado.
add_library(mamba_kernel SHARED mamba_ssm_op.cc mamba_ssm_kernel.cu)

# Opción Robusta - Metadatos y Optimización por Lenguaje (Generator Expressions):
# 1. -O3: Máxima agresividad matemática en Kernel.
# 2. -lineinfo: Exponemos información de debug a los profilers "ncu"/"nsys" sin costo real.
# 3. Forzamos código dependiente de posición independizado (-fPIC) para la capa del Host.
target_compile_options(mamba_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC -O2>
)

# Opción Robusta - Ensamblado de Arquitectura Gráfica Nivel Hardware
# 75 = Turing (Tesla T4 en Colab gratuito)
# 80 = Ampere (A100 en Colab Pro)
# ¿Por qué? Prevenimos que el nvcc genere un código intermedio universal PTX (PseudoEnsamblador) y obligamos  
# a que en su lugar inyecte código en Ensamblador SASS nativo. Esto extirpa las severas sobrecargas por JIT compilation
# en tiempo de ejecución para cada inferencia primeriza (TensorFlow Warmup Stalls).
set_target_properties(mamba_kernel PROPERTIES CUDA_ARCHITECTURES "75;80")

# Inyectamos los flags nativos de TF usando la API moderna de CMake.
# separate_arguments() convierte el string a lista CMake; COMPILE_LANGUAGE:CXX
# asegura que los flags de ABI de TF (-D_GLIBCXX_USE_CXX11_ABI=0) se apliquen
# exclusivamente al compilador C++ Host (g++) y no contaminen nvcc.
separate_arguments(TF_COMPILE_FLAGS_LIST NATIVE_COMMAND "${TF_COMPILE_FLAGS_STR}")
separate_arguments(TF_LINK_FLAGS_LIST NATIVE_COMMAND "${TF_LINK_FLAGS_STR}")

target_compile_options(mamba_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TF_COMPILE_FLAGS_LIST}>
)
target_link_options(mamba_kernel PRIVATE ${TF_LINK_FLAGS_LIST})
