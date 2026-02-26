cmake_minimum_required(VERSION 3.14)
project(MambaVision_TF_Kernel)

# Asegurar que se usará C++17 como requiere TF en configuraciones modernas
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar Python y el paquete de TensorFlow
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Ejecutar un script para obtener los flags de TF (Include y Link)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TF_INC
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ejecutar script para obtener el flag de compilacion natural y exacto del sistema (ej: el ABI que usa Colab)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ejecutar un script para obtener los flags exactos de linkado
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Añadir directorio de inclusión de headers de Tensorflow
include_directories(${TF_INC})

# Crear la biblioteca compartida .so a partir del código C++ de Mamba
add_library(mamba_kernel SHARED mamba_ssm_op.cc)

# Nota: Omitimos set_target_properties(PREFIX "") para que CMake 
# empaquete naturalmente como "libmamba_kernel.so" en Linux.

# Eliminar el override de ABI y RTTI:
# Descubrimos gracias a la prueba de compilación cruda con g++ que Colab 
# utiliza internamente ABI=1 y C++ estándar. Insertar -fno-rtti o forzar ABI=0 rompe el linker.
target_compile_options(mamba_kernel PRIVATE -fPIC -O2)

# Añadimos limpiamente todos los flags nativos de TF al target sin alterarlos
set_target_properties(mamba_kernel PROPERTIES COMPILE_FLAGS "${TF_COMPILE_FLAGS_STR}")
set_target_properties(mamba_kernel PROPERTIES LINK_FLAGS "${TF_LINK_FLAGS_STR}")
