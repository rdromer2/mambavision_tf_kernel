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
    // Calculamos el índice global lineal (astronómico) y el índice local del hilo dentro del bloque
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; // Índice de 0 a 255
    
    // --- FASE 3.5: RESERVA ESTÁTICA EN MEMORIA COMPARTIDA (SRAM L1) ---
    // ¿Por qué tamaño fijo 256? Al lanzarlo desde C++ fijamos `threads_per_block = 256`.
    // La memoria __shared__ la ven en comunión exclusivamente los 256 hilos de este bloque físico.
    __shared__ float s_u[256];
    __shared__ float s_h[256];

    // --- CARGA COOPERATIVA DE VRAM A SRAM ---
    // Si el hilo global está dentro de un dato válido, carga el dato físico de VRAM
    if (idx < total_elements) {
        s_u[tid] = u_in[idx];
        s_h[tid] = 0.0f; // Inicialización temporal del Hidden State de Mamba
    } else {
        // Purgado de las posiciones de caché muertas en los bordes para el bloque final (Padding)
        s_u[tid] = 0.0f;
        s_h[tid] = 0.0f;
    }

    // --- BARRERA FÍSICA INQUEBRANTABLE ---
    // ¿Por qué está aquí suelto y fuera del IF? 
    // Un __syncthreads() "if statement" divergente provoca un DEADLOCK, porque los hilos
    // que entran al if esperan infinitamente a los que se fueron por el else.
    // Aquí bloqueamos el reloj del SM hasta certificar que s_u[] de [0 a 255] se ha rellenado al completo.
    __syncthreads();

    // <--- MARCADOR FUTURO: MATEMÁTICAS SELECTIVE SCAN RECURRENTES AQUÍ --->
    
    // --- ESCRITURA VRAM DESCENDENTE TEMPORAL ---
    // Ya seguros de que SRAM está nutrida y sincronizada matemáticamente, 
    // volcamos los resultados estancados de nuevo a la memoria visible por el sistema host
    if (idx < total_elements) {
        // En Fase 3 lo leíamos de `u_in`. Ahora es `s_u` (que multiplicaremos de paso para test visual)
        out[idx] = s_u[tid] * 2.0f;
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
