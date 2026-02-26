#include <cuda_runtime.h>
#include <stdio.h>

// Macro robusta para capturar errores de CUDA (síncronos y asíncronos)
// ¿Por qué? Las llamadas de CUDA son asíncronas respecto a la CPU.
// Si lanzamos un kernel que genera un "segmentation fault" (acceso ilegal de memoria en VRAM),
// la falla no explota en la línea del kernel en el host, sino más adelante o al leer resultados.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

// 1. El Kernel de CUDA (Se ejecuta en la VRAM de la gráfica)
// El cualificador __global__ indica que la función corre en GPU y es invocable desde CPU.
__global__ void MambaSelectiveScanKernel(const float* u_in, float* out, int total_elements) {
    // Calculamos el índice global lineal del hilo actual a través de la topología CUDA.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verificamos no salirnos de los límites del arreglo (crítico en memoria de GPU).
    if (idx < total_elements) {
        // En esta Fase 3, usamos una operación dummy de prueba para verificar
        // el enlazado completo (TensorFlow -> C++ -> GPU) antes de inyectar las ecuaciones reales.
        out[idx] = u_in[idx] * 2.0f;
    }
}

// 2. La función *Launcher* en C++ estándar
// ¿Por qué extern "C"? Evitamos el C++ "Name Mangling". Esto garantiza que el enlazador g++
// de TensorFlow encuentre el nombre exacto de la función exportada desde nvcc sin sufijos raros.
extern "C" cudaError_t LaunchMambaSelectiveScan(const float* d_in, float* d_out, int batch_size, int seq_len, int d_model) {
    int total_elements = batch_size * seq_len * d_model;

    // Configuración de la topología de ejecución:
    // Ocupación estándar de hilos por bloque: 256 (Típicamente óptimo en la arquitectura SIMT)
    int threads_per_block = 256;
    // Cálculo de la cuadrícula: redondeo siempre al alza por bloque para cubrir los sobrantes.
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Lanzamos el kernel (sintaxis triple chevron específica del compilador nvcc)
    MambaSelectiveScanKernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, total_elements);

    // Captura estricta de errores (Opción Robusta)
    // 1. Chequear errores inmediatos en el lanzamiento del kernel (ej. dimensiones inválidas)
    CUDA_CHECK(cudaGetLastError());
    
    // 2. Forzar sincronización Host <-> Device
    // Bloquea temporalmente el flujo host para esperar a que los hilos VRAM acaben.
    // Garantiza emitir un error ahora y no corromper estados en Python.
    CUDA_CHECK(cudaDeviceSynchronize());

    return cudaSuccess;
}
