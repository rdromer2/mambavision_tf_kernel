# 🛠️ MambaVision TF Kernel: Project Dev Context

Documento técnico de esquema de componentes para enrutamiento rápido del proyecto. Si estás implementando nuevo código, revisa estas invariables arquitectónicas.

## 1. Módulo 1: Pipeline de Construcción (`CMakeLists.txt`)
- **Sistema de dependencias:** Gestionado enteramente por CMake delegando subprocesos de Python para extraer las rutas e includes del framework interno de TF (`tf.sysconfig`).
- **Orquestador Híbrido:**
  - Código `.cc` pasa por `g++` (modo C++17, obligatorio).
  - Código `.cu` pasa por `nvcc`.
- **Estrategia de Banderas de Compilación (Opción Robusta):**
  - **Host:** `-fPIC` (Position Independent Code, para `.so` dinámico) y `-O2` por estabilidad.
  - **Device/CUDA:** `-O3` (agresividad máxima iterativa) y `-lineinfo` para perfiles Nsight (`ncu`/`nsys`) sin sobrecarga temporal.
  - **Ensamblador Target:** Fijado a `CUDA_ARCHITECTURES "75;80"` (T4 y A100). Evita la generación intermedia genérica (PTX JIT) previniendo las demoras masivas de la primera pasada ("warmup stall").

## 2. Módulo 2: Driver del Kernel C++ (`mamba_ssm_op.cc`)
- **Registro API (`REGISTER_OP`):** Define el bloque de interface a Python (`MambaSelectiveScan`). Implementa *shape inference*, la cual se espera que por lo general preserve la dimensionalidad de salida (`out_tensor_shape == in_tensor_shape`).
- **El Centinela (Clase `OpKernel`):**
  - Instanciada e inyectada explícitamente a Device GPU (`REGISTER_KERNEL_BUILDER(Name("MambaSelectiveScan").Device(DEVICE_GPU))`), lo cual obliga a TensorFlow a alojar implícitamente la memoria en VRAM, eliminando las copias `cudaMemcpyHostToDevice`.
  - **Procedimiento `Compute`:** Requisitos formales fuertes (`OP_REQUIRES` a dimensionalidad correcta), lectura de sub-dimensiones tensoras.
  - **Puente Punteros:** Extrae `input_tensor.flat<float>().data()` (Dirección C subyacente cruda de memoria gráfica).
- **Ejecución Asíncrona:** Llama al encapsulador CUDA externo (definido via `extern "C"`) el cual orquestará los *Blocks* y *Threads*, retornando el status de CUDA para ser escalado como excepción estandarizada C++ a Python.

## 3. Módulo 3: Ejecución Física CUDA (`mamba_ssm_kernel.cu`)
*(Actual en Fase de Desarrollo)*
- Expone obligatoriamente un conector C compatible: `extern "C" int LaunchMambaSelectiveScan(...)`.
- Responsable de la asignación lógica de `dim3 blocks, threads`, transferencia controlada a `__shared__` memory para computación acelerada, y bloqueos por barrera (`__syncthreads()`).

## 4. Módulo 4: Pipeline CI/CD Efímero (`compilador.ipynb`)
El entorno de integración asíncrono se realiza mediante celdas en Colab.
1. Actualización sincronizada via GitHub API (`git pull`).
2. Delega explícitamente toda macro compilación al binario local haciendo un hard reset `/build` -> `cmake ..` -> `make -j`.
3. **Parche Crítico de Binding (Linker Resolver):** Dado el ecosistema precompilado *out-of-tree* en Colab, es imperativo interconectar dinámicamente los símbolos no resueltos (`undefined symbol: _ZTVN10...`) inyectando vía alto nivel:
   ```python
   import ctypes
   ctypes.CDLL('/usr/local/lib/python3.12/dist-packages/tensorflow/libtensorflow_framework.so.2', ctypes.RTLD_GLOBAL)
   ```
   Antes de llamar a `tf.load_op_library("build/libmamba_kernel.so")`.
4. Pruebas iterativas: Inyección explícita de tensores de validación bajo scope forzado `with tf.device('/GPU:0'):` y comprobación asercional matemática `np.allclose`.
