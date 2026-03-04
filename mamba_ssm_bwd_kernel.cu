#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Macro robusta para capturar errores de CUDA (síncronos y asíncronos)
// Réplica exacta de la macro del Forward Pass para consistencia de diagnóstico.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA BWD Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

// ============================================================================
// MÓDULO MATEMÁTICO AISLADO (__device__)
// ============================================================================
// Réplica EXACTA de las funciones del Forward Pass (mamba_ssm_kernel.cu).
// Esto garantiza que la discretización y la acumulación del estado oculto
// en el Backward usen exactamente los mismos binarios que el Forward,
// eliminando discrepancias de precisión en punto flotante entre pasadas.

// Discretización paramétrica con Zero-Order Hold (ZOH)
// A_bar = exp(A * delta)
__device__ inline float discretize_A(const float A, const float delta) {
    return expf(A * delta);
}

// Computación iterativa del estado oculto recurrente
// h_t = A_bar * h_{t-1} + B_bar * x_t
// fmaf() ejecuta la acumulación en una sola instrucción atómica de hardware.
__device__ inline float compute_scan_step(const float A_bar, const float h_prev, const float B, const float x, const float delta) {
    float B_bar = B * delta;
    return fmaf(A_bar, h_prev, B_bar * x);
}

// ============================================================================
// KERNEL BACKWARD: DOBLE BARRIDO CON VRAM WORKSPACE
// ============================================================================
// Topología Hilo-por-Canal idéntica al Forward.
// Cada hilo ejecuta dos barridos secuenciales sobre el eje temporal:
//   1. Barrido Forward (Recomputación): Recalcula h_t y lo vuelca al Workspace VRAM.
//   2. Barrido Backward (BPTT): Lee h_t del Workspace, calcula los 5 gradientes.
//
// Reglas de concurrencia:
//   - du, dDelta, dB, dC: Escritura directa. Cada hilo posee dirección VRAM exclusiva.
//   - dA: atomicAdd obligatorio. A es [d_model] (compartido entre batches).

__global__ void MambaSelectiveScanBwdKernel(
    // 6 inputs (lectura desde VRAM)
    const float* dy, const float* u, const float* delta,
    const float* A, const float* B, const float* C,
    // 5 outputs de gradiente (escritura a VRAM)
    float* du_out, float* dDelta_out, float* dA_out, float* dB_out, float* dC_out,
    // Workspace temporal (lectura/escritura a VRAM)
    float* h_workspace,
    // Dimensiones geométricas
    int batch_size, int seq_len, int d_model) {

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_channels_total = batch_size * d_model;

    if (global_idx >= num_channels_total) return;

    // Desanidar tensor: mapeo lógico [batch, d_model]
    int b = global_idx / d_model;
    int d = global_idx % d_model;

    // A es [d_model]: estático por canal. Lectura única, retenido en registro.
    const float local_A = A[d];

    // ====================================================================
    // BARRIDO 1: FORWARD (Recomputación de h_t → Workspace VRAM)
    // ====================================================================
    // Objetivo: Recalcular la traza completa de estados ocultos h_t
    // y almacenarla en el tensor Workspace para que el barrido backward
    // pueda leer h_t y h_{t-1} sin recomputación cuadrática O(L²).
    float h_prev = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        // Stride coalescente: hilos vecinos (d, d+1...) leen posiciones contiguas
        int idx = (b * seq_len * d_model) + (t * d_model) + d;

        // Lectura de parámetros de VRAM (mismos que el Forward)
        float x_t         = u[idx];
        float local_delta  = delta[idx];
        float local_B      = B[idx];

        // Discretización dinámica por timestep
        float A_bar = discretize_A(local_A, local_delta);

        // Acumulación del estado oculto
        h_prev = compute_scan_step(A_bar, h_prev, local_B, x_t, local_delta);

        // Volcar h_t al Workspace en VRAM.
        // El stride es idéntico al de u: [batch, seq_len, d_model] C-contiguous.
        // Hilos vecinos (d, d+1) escriben posiciones contiguas → coalescente.
        h_workspace[idx] = h_prev;
    }

    // ====================================================================
    // BARRIDO 2: BACKWARD (BPTT - Backpropagation Through Time)
    // ====================================================================
    // Objetivo: Recorrer el eje temporal en reversa, aplicando la regla de
    // la cadena para calcular las 5 derivadas parciales analíticas.
    //
    // Variables de propagación en registros privados del hilo:
    //   dh_next:    Gradiente del estado oculto propagado desde el futuro (t+1 → t)
    //   next_A_bar: Ā del timestep t+1, necesario para propagar dh
    //   dA_local:   Acumulador del gradiente de A sobre todos los timesteps

    float dh_next    = 0.0f;   // Condición terminal: dh_{seq_len} = 0
    float next_A_bar = 0.0f;   // No hay timestep posterior al último
    float dA_local   = 0.0f;   // Acumulador en registro privado (evita condiciones de carrera)

    for (int t = seq_len - 1; t >= 0; --t) {
        int idx = (b * seq_len * d_model) + (t * d_model) + d;

        // ----------------------------------------------------------------
        // LECTURA DE PARÁMETROS DESDE VRAM
        // ----------------------------------------------------------------
        // Los tensores originales (u, delta, B, C) fueron retenidos por TF
        // desde el Forward Pass. Los releemos directamente de VRAM.
        float x_t         = u[idx];
        float local_delta  = delta[idx];
        float local_B      = B[idx];
        float local_C      = C[idx];
        float dy_t         = dy[idx];

        // Recomputar A_bar (1 instrucción expf en silicio, coste despreciable)
        float A_bar = discretize_A(local_A, local_delta);

        // ----------------------------------------------------------------
        // LECTURA DE ESTADOS DESDE WORKSPACE VRAM
        // ----------------------------------------------------------------
        // h_t:     El estado oculto en el timestep actual
        // h_prev:  El estado oculto en t-1 (h_{-1} = 0 como caso base)
        float h_t = h_workspace[idx];
        float h_prev_val = (t > 0)
            ? h_workspace[(b * seq_len * d_model) + ((t - 1) * d_model) + d]
            : 0.0f;

        // ----------------------------------------------------------------
        // CÁLCULO DE GRADIENTES POR REGLA DE LA CADENA
        // ----------------------------------------------------------------

        // dC_t = dy_t · h_t (directo desde y_t = C_t · h_t)
        // Escritura directa: cada hilo posee idx exclusivo.
        dC_out[idx] = dy_t * h_t;

        // dh_t = dy_t · C_t + dh_{t+1} · Ā_{t+1}
        // La contribución del futuro usa next_A_bar (Ā del paso t+1).
        float dh_t = dy_t * local_C + dh_next * next_A_bar;

        // du_t = dh_t · B̄_t = dh_t · B_t · Δ_t
        // Escritura directa: idx es exclusivo por hilo.
        du_out[idx] = dh_t * local_B * local_delta;

        // dB_t = dh_t · u_t · Δ_t
        // Escritura directa: idx es exclusivo por hilo.
        dB_out[idx] = dh_t * x_t * local_delta;

        // dΔ_t = dh_t · h_{t-1} · A_d · Ā_t  +  dh_t · B_t · u_t
        //        ↑ contribución de exp(A·Δ)        ↑ contribución de B·Δ
        // Escritura directa: idx es exclusivo por hilo.
        dDelta_out[idx] = dh_t * h_prev_val * local_A * A_bar
                        + dh_t * local_B * x_t;

        // dA: Acumulación iterativa en registro privado.
        // dA_d += dh_t · h_{t-1} · Δ_t · Ā_t
        // NO se escribe a VRAM aquí; se vuelca una sola vez al final.
        dA_local += dh_t * h_prev_val * local_delta * A_bar;

        // Propagar al timestep anterior: guardar Ā actual para el siguiente
        // paso del bucle (que procesará t-1 y necesitará Ā_t).
        next_A_bar = A_bar;
        dh_next = dh_t;
    }

    // ====================================================================
    // VOLCADO FINAL DE dA: OPERACIÓN ATÓMICA OBLIGATORIA
    // ====================================================================
    // A es [d_model]: compartido entre todos los elementos del batch.
    // Múltiples hilos de diferentes batches (b=0, b=1, ...) escriben al
    // mismo dA_out[d]. atomicAdd serializa el acceso, evitando la condición
    // de carrera de escritura concurrente.
    atomicAdd(&dA_out[d], dA_local);
}

// ============================================================================
// FUNCIÓN LAUNCHER EXPORTABLE A TENSORFLOW
// ============================================================================
// extern "C" evita el C++ Name Mangling, garantizando que el enlazador g++
// de TensorFlow encuentre el nombre exacto exportado desde nvcc.
extern "C" cudaError_t LaunchMambaSelectiveScanBwd(
    const float* dy, const float* u, const float* delta,
    const float* A, const float* B, const float* C,
    float* du_out, float* dDelta_out, float* dA_out, float* dB_out, float* dC_out,
    float* h_workspace,
    int batch_size, int seq_len, int d_model) {

    // Inicialización pre-kernel de dA_out (escala: [d_model])
    // Necesario porque atomicAdd en el kernel acumulará desde cero.
    cudaMemset(dA_out, 0, d_model * sizeof(float));

    int num_channels_total = batch_size * d_model;
    int threads_per_block = 256;
    int blocks_per_grid = (num_channels_total + threads_per_block - 1) / threads_per_block;

    MambaSelectiveScanBwdKernel<<<blocks_per_grid, threads_per_block>>>(
        dy, u, delta, A, B, C,
        du_out, dDelta_out, dA_out, dB_out, dC_out,
        h_workspace,
        batch_size, seq_len, d_model);

    // Contención estricta diagnóstica (misma estrategia que Forward)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return cudaSuccess;
}
