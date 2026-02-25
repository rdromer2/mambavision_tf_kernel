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

# Crear la biblioteca compartida .so a partir del código C++
add_library(zero_out SHARED zero_out.cc)

# Evitar el prefijo "lib" convencional en Linux y el .so
set_target_properties(zero_out PROPERTIES PREFIX "" SUFFIX ".so")

# === FIX DEFINITIVO DE ABI + RTTI ===
# Desactivamos el "RTTI" (-fno-rtti) del compilador de Colab, porque los paquetes de TensorFlow alojados en la nube están compilados ciegos y ocultan la TypeInformation virtual.
target_compile_options(zero_out PRIVATE -fno-rtti -fno-exceptions -fPIC -O2)

# Colab's tf.sysconfig devuelve por defecto -D_GLIBCXX_USE_CXX11_ABI=1, lo cual es incompatible
# con el binario pre-compilado de pip que requiere 0. 
# Lo limpiamos usando regex en CMake para evitar advertencias de "redefined":
string(REGEX REPLACE "-D_GLIBCXX_USE_CXX11_ABI=[0-9]" "" TF_COMPILE_FLAGS_STR_CLEAN "${TF_COMPILE_FLAGS_STR}")

# Finalmente lo re-inyectamos como 0 explícitamente:
set(TF_COMPILE_FLAGS_FINAL "${TF_COMPILE_FLAGS_STR_CLEAN} -D_GLIBCXX_USE_CXX11_ABI=0")

# Añadimos limpiamente todos los flags al target
set_target_properties(zero_out PROPERTIES COMPILE_FLAGS "${TF_COMPILE_FLAGS_FINAL}")
set_target_properties(zero_out PROPERTIES LINK_FLAGS "${TF_LINK_FLAGS_STR}")
