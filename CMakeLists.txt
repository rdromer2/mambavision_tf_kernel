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

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TF_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ejecutar un script para obtener el flag ABI correcto que utiliza la instalación de TF actual
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ejecutar un script para obtener los flags exactos de linkado
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Añadir directorio de inclusión de headers de Tensorflow
include_directories(${TF_INC})

# Crear la biblioteca compartida .so a partir del código C++
add_library(zero_out SHARED zero_out.cc)

# Evitar el prefijo "lib" convencional en Linux
set_target_properties(zero_out PROPERTIES PREFIX "")
# Especificar que el SO será un módulo cargable dinámicamente (.so)
set_target_properties(zero_out PROPERTIES SUFFIX ".so")

# Configurar flags de compilación. En lugar de forzar ABI=1, usamos los flags exactos provistos por tf.sysconfig
target_compile_options(zero_out PRIVATE -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(zero_out PROPERTIES COMPILE_FLAGS ${TF_COMPILE_FLAGS})

# Configurar los flags de enlazado provistos por tf.sysconfig para agarrar libtensorflow_framework.so.2 correcto
set_target_properties(zero_out PROPERTIES LINK_FLAGS ${TF_LINK_FLAGS})
