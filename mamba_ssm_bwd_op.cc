#include <cuda_runtime_api.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// ============================================================================
// DECLARACIÓN DEL LAUNCHER CUDA BACKWARD
// ============================================================================
// extern "C" garantiza que g++ encuentre el símbolo exacto exportado por nvcc
// sin C++ Name Mangling. El tipo de retorno es cudaError_t (int bajo el capó).
extern "C" int LaunchMambaSelectiveScanBwd(
    const float* dy, const float* u, const float* delta,
    const float* A, const float* B, const float* C,
    float* du_out, float* dDelta_out, float* dA_out, float* dB_out, float* dC_out,
    float* h_workspace,
    int batch_size, int seq_len, int d_model);

// ============================================================================
// 1. REGISTRO DE LA INTERFAZ HACIA PYTHON
// ============================================================================
// La operación de gradiente recibe 6 inputs (el gradiente dy + los 5 tensores
// originales del forward) y produce 5 outputs (los gradientes parciales).
// NOTA: Usar nombres descriptivos en minúsculas para evitar el conflicto con
// las variables de tipo reservadas de TF (A, B, C, T).
REGISTER_OP("MambaSelectiveScanGrad")
    .Input("dy: float")         // [batch, seq_len, d_model]  - Gradiente de la pérdida
    .Input("u: float")          // [batch, seq_len, d_model]  - Entrada original
    .Input("delta: float")      // [batch, seq_len, d_model]  - Step size original
    .Input("a_param: float")    // [d_model]                  - Matriz de estado original
    .Input("b_param: float")    // [batch, seq_len, d_model]  - Matriz entrada original
    .Input("c_param: float")    // [batch, seq_len, d_model]  - Matriz salida original
    .Output("du: float")        // [batch, seq_len, d_model]
    .Output("d_delta: float")   // [batch, seq_len, d_model]
    .Output("d_a: float")       // [d_model]
    .Output("d_b: float")       // [batch, seq_len, d_model]
    .Output("d_c: float")       // [batch, seq_len, d_model]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // du, d_delta, d_b, d_c tienen la misma shape que u (input 1)
        c->set_output(0, c->input(1));  // du     = shape(u)
        c->set_output(1, c->input(1));  // d_delta = shape(u)
        c->set_output(3, c->input(1));  // d_b    = shape(u)
        c->set_output(4, c->input(1));  // d_c    = shape(u)
        // d_a tiene la misma shape que a_param (input 3)
        c->set_output(2, c->input(3));  // d_a    = shape(A)
        return absl::OkStatus();
    });

// ============================================================================
// 2. LA CLASE DEL KERNEL (Orquestador Host del Backward)
// ============================================================================
class MambaSelectiveScanGradOp : public OpKernel {
public:
    explicit MambaSelectiveScanGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // ====================================================================
        // CAPTURA DE LOS 6 TENSORES DE ENTRADA
        // ====================================================================
        const Tensor& dy_tensor    = context->input(0);  // dy:    [batch, seq_len, d_model]
        const Tensor& u_tensor     = context->input(1);  // u:     [batch, seq_len, d_model]
        const Tensor& delta_tensor = context->input(2);  // delta: [batch, seq_len, d_model]
        const Tensor& A_tensor     = context->input(3);  // A:     [d_model]
        const Tensor& B_tensor     = context->input(4);  // B:     [batch, seq_len, d_model]
        const Tensor& C_tensor     = context->input(5);  // C:     [batch, seq_len, d_model]

        // ====================================================================
        // VALIDACIÓN ESTRICTA DIMENSIONAL
        // ====================================================================
        // dy debe ser 3D
        OP_REQUIRES(context, dy_tensor.dims() == 3,
                    errors::InvalidArgument("dy requiere tensor 3D. Recibido: ",
                                            dy_tensor.dims(), "D"));

        // u debe ser 3D
        OP_REQUIRES(context, u_tensor.dims() == 3,
                    errors::InvalidArgument("u requiere tensor 3D. Recibido: ",
                                            u_tensor.dims(), "D"));

        const int batch_size = u_tensor.dim_size(0);
        const int seq_len    = u_tensor.dim_size(1);
        const int d_model    = u_tensor.dim_size(2);

        // dy debe coincidir con u en shape
        OP_REQUIRES(context, dy_tensor.dim_size(0) == batch_size &&
                             dy_tensor.dim_size(1) == seq_len &&
                             dy_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("dy shape debe coincidir con u: [",
                                            batch_size, ", ", seq_len, ", ", d_model, "]"));

        // delta debe ser 3D con misma shape que u
        OP_REQUIRES(context, delta_tensor.dims() == 3 &&
                             delta_tensor.dim_size(0) == batch_size &&
                             delta_tensor.dim_size(1) == seq_len &&
                             delta_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("delta shape debe coincidir con u"));

        // A debe ser 1D: [d_model]
        OP_REQUIRES(context, A_tensor.dims() == 1 &&
                             A_tensor.dim_size(0) == d_model,
                    errors::InvalidArgument("A requiere tensor 1D [d_model=", d_model,
                                            "]. Recibido: dims=", A_tensor.dims()));

        // B debe ser 3D con misma shape que u
        OP_REQUIRES(context, B_tensor.dims() == 3 &&
                             B_tensor.dim_size(0) == batch_size &&
                             B_tensor.dim_size(1) == seq_len &&
                             B_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("B shape debe coincidir con u"));

        // C debe ser 3D con misma shape que u
        OP_REQUIRES(context, C_tensor.dims() == 3 &&
                             C_tensor.dim_size(0) == batch_size &&
                             C_tensor.dim_size(1) == seq_len &&
                             C_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("C shape debe coincidir con u"));

        // ====================================================================
        // ALOCAR 5 TENSORES DE SALIDA (Gradientes → retornan a Python)
        // ====================================================================
        Tensor* du_tensor = nullptr;
        Tensor* dDelta_tensor = nullptr;
        Tensor* dA_tensor = nullptr;
        Tensor* dB_tensor = nullptr;
        Tensor* dC_tensor = nullptr;

        // du, dDelta, dB, dC: misma shape que u [batch, seq_len, d_model]
        OP_REQUIRES_OK(context, context->allocate_output(0, u_tensor.shape(), &du_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, u_tensor.shape(), &dDelta_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(3, u_tensor.shape(), &dB_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(4, u_tensor.shape(), &dC_tensor));

        // dA: misma shape que A [d_model]
        OP_REQUIRES_OK(context, context->allocate_output(2, A_tensor.shape(), &dA_tensor));

        // ====================================================================
        // WORKSPACE TEMPORAL: Tensor de estados h_t
        // ====================================================================
        // allocate_temp() crea un tensor que SOLO existe durante Compute().
        // TF lo destruye automáticamente al salir — Python nunca lo ve.
        // Shape: [batch, seq_len, d_model] (misma geometría que u).
        Tensor h_workspace;
        OP_REQUIRES_OK(context,
            context->allocate_temp(DT_FLOAT, u_tensor.shape(), &h_workspace));

        // ====================================================================
        // EXTRACCIÓN DE PUNTEROS VRAM
        // ====================================================================
        // 6 punteros de entrada (lectura)
        const float* d_dy_ptr    = dy_tensor.flat<float>().data();
        const float* d_u_ptr     = u_tensor.flat<float>().data();
        const float* d_delta_ptr = delta_tensor.flat<float>().data();
        const float* d_A_ptr     = A_tensor.flat<float>().data();
        const float* d_B_ptr     = B_tensor.flat<float>().data();
        const float* d_C_ptr     = C_tensor.flat<float>().data();

        // 5 punteros de salida (escritura)
        float* d_du_ptr     = du_tensor->flat<float>().data();
        float* d_dDelta_ptr = dDelta_tensor->flat<float>().data();
        float* d_dA_ptr     = dA_tensor->flat<float>().data();
        float* d_dB_ptr     = dB_tensor->flat<float>().data();
        float* d_dC_ptr     = dC_tensor->flat<float>().data();

        // 1 puntero del Workspace (lectura/escritura interna)
        float* d_workspace_ptr = h_workspace.flat<float>().data();

        // ====================================================================
        // INICIALIZACIÓN DE dA A CERO (target de atomicAdd)
        // ====================================================================
        // Antes de que el kernel haga atomicAdd sobre dA, el tensor debe
        // contener ceros. cudaMemset es síncrono respecto al stream default.
        cudaMemset(d_dA_ptr, 0, d_model * sizeof(float));

        // ====================================================================
        // INVOCACIÓN AL DOMINIO NVIDIA/CUDA
        // ====================================================================
        int cuda_status = LaunchMambaSelectiveScanBwd(
            d_dy_ptr, d_u_ptr, d_delta_ptr, d_A_ptr, d_B_ptr, d_C_ptr,
            d_du_ptr, d_dDelta_ptr, d_dA_ptr, d_dB_ptr, d_dC_ptr,
            d_workspace_ptr,
            batch_size, seq_len, d_model);

        OP_REQUIRES(context, cuda_status == 0,
                    errors::Internal("GPU Ejecución falló en LaunchMambaSelectiveScanBwd. "
                                     "Error CUDA: ", cuda_status));
    }
};

// ============================================================================
// 3. REGISTRAR EL KERNEL PARA LA GPU
// ============================================================================
REGISTER_KERNEL_BUILDER(Name("MambaSelectiveScanGrad").Device(DEVICE_GPU), MambaSelectiveScanGradOp);
