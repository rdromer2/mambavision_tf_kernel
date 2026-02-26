#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Macro robusta para capturar errores de CUDA (síncronos y asíncronos)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

// --- MÓDULO MATEMÁTICO AISLADO (__device__) ---
// Separación logística-matemática estricta. Obliga a que tanto el Forward Pass
// como el futuro Backward Pass (Fase 4) invoquen exactamente los mismos binarios,
// eliminando cualquier discrepancia de precisión en punto flotante.

// Discretización paramétrica con Zero-Order Hold (ZOH)
// A_bar = exp(A * delta)
__device__ inline float discretize_A(const float A, const float delta) {
    return expf(A * delta);
}

// Computación iterativa del estado oculto recurrente
// h_t = A_bar * h_{t-1} + B_bar * x_t
// Utiliza fmaf() (Fused Multiply-Add) para ejecutar la acumulación en una sola
// instrucción atómica de hardware, limitando la cascada de redondeo IEEE 754.
__device__ inline float compute_scan_step(const float A_bar, const float h_prev, const float B, const float x, const float delta) {
    float B_bar = B * delta;
    return fmaf(A_bar, h_prev, B_bar * x);
}

// --- MÓDULO PRINCIPAL DE FLUJO GPU (FORWARD PASS) ---
// Mapeo Hilo-por-Canal. Flujo directo VRAM -> Registro -> VRAM.
// Cada hilo se asigna a un canal (d_model) de un elemento del batch.
// Dentro del kernel, el hilo ejecuta un bucle secuencial sobre seq_len,
// propagando el estado oculto h_prev en su registro ultrarrápido privado.
__global__ void MambaSelectiveScanKernel(const float* u_in, float* out, int batch_size, int seq_len, int d_model) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_channels_total = batch_size * d_model;

    if (global_idx >= num_channels_total) return;

    // Desanidar tensor: mapeo lógico [batch, d_model]
    int b = global_idx / d_model;
    int d = global_idx % d_model;

    // Marcadores paramétricos estáticos de validación (serán reemplazados por tensores
    // dinámicos proyectados desde Python en fases posteriores de integración)
    const float local_delta = 0.1f;
    const float local_A = -1.0f;
    const float local_B = 1.0f;

    // Pre-cálculo de discretización cautivo en registro del Thread
    float A_bar = discretize_A(local_A, local_delta);

    // Estado oculto causal en registro ultrarrápido (resuelve dependencia h_{t-1})
    float h_prev = 0.0f;

    // Propagación causal secuencial a lo largo del eje temporal
    for (int t = 0; t < seq_len; ++t) {
        // Stride coalescente: hilos vecinos (d, d+1...) leen posiciones contiguas
        int tensor_idx = (b * seq_len * d_model) + (t * d_model) + d;

        // VRAM -> Registro
        float x_t = u_in[tensor_idx];

        // Computación atómica FMA sobre registro privado
        h_prev = compute_scan_step(A_bar, h_prev, local_B, x_t, local_delta);

        // Registro -> VRAM
        out[tensor_idx] = h_prev;
    }
}

// Función Launcher en C++ estándar exportable al wrapper TensorFlow
// extern "C" evita el C++ Name Mangling, garantizando que el enlazador g++
// de TensorFlow encuentre el nombre exacto de la función exportada desde nvcc.
extern "C" cudaError_t LaunchMambaSelectiveScan(const float* d_in, float* d_out, int batch_size, int seq_len, int d_model) {
    int num_channels_total = batch_size * d_model;
    int threads_per_block = 256;
    int blocks_per_grid = (num_channels_total + threads_per_block - 1) / threads_per_block;

    MambaSelectiveScanKernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, batch_size, seq_len, d_model);

    // Contención estricta diagnóstica
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return cudaSuccess;
}
