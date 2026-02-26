#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// Declaración de la función externa en CUDA que invoca al kernel de NVIDIA.
// Retorna cudaError_t (como un int) bajo el capó.
// Fase 4: Firma expandida con punteros a los tensores paramétricos reales.
extern "C" int LaunchMambaSelectiveScan(
    const float* d_in, float* d_out,
    const float* d_delta, const float* d_A, const float* d_B, const float* d_C,
    int batch_size, int seq_len, int d_model);

// 1. Registro de la interfaz hacia Python
REGISTER_OP("MambaSelectiveScan")
    .Input("u: float")       // [batch, seq_len, d_model] - Entrada secuencial
    .Input("delta: float")   // [batch, seq_len, d_model] - Step size de discretización
    .Input("A: float")       // [d_model]                 - Matriz de estado (estática por canal)
    .Input("B: float")       // [batch, seq_len, d_model] - Matriz de entrada (input-dependent)
    .Input("C: float")       // [batch, seq_len, d_model] - Matriz de salida (input-dependent)
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    });

// 2. La clase del Kernel (El Centinela / Orquestador Host)
class MambaSelectiveScanOp : public OpKernel {
public:
    explicit MambaSelectiveScanOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // ====================================================================
        // CAPTURA DE TENSORES DE ENTRADA
        // ====================================================================
        const Tensor& input_tensor = context->input(0);  // u:     [batch, seq_len, d_model]
        const Tensor& delta_tensor = context->input(1);   // delta: [batch, seq_len, d_model]
        const Tensor& A_tensor     = context->input(2);   // A:     [d_model]
        const Tensor& B_tensor     = context->input(3);   // B:     [batch, seq_len, d_model]
        const Tensor& C_tensor     = context->input(4);   // C:     [batch, seq_len, d_model]

        // ====================================================================
        // VALIDACIÓN ESTRICTA DIMENSIONAL
        // ====================================================================
        // u debe ser 3D: [batch, seq_len, d_model]
        OP_REQUIRES(context, input_tensor.dims() == 3,
                    errors::InvalidArgument("u requiere tensor 3D [batch, seq_len, d_model]. Recibido: ",
                                            input_tensor.dims(), "D"));

        const int batch_size = input_tensor.dim_size(0);
        const int seq_len    = input_tensor.dim_size(1);
        const int d_model    = input_tensor.dim_size(2);

        // delta debe ser 3D con misma shape que u
        OP_REQUIRES(context, delta_tensor.dims() == 3,
                    errors::InvalidArgument("delta requiere tensor 3D. Recibido: ",
                                            delta_tensor.dims(), "D"));
        OP_REQUIRES(context, delta_tensor.dim_size(0) == batch_size &&
                             delta_tensor.dim_size(1) == seq_len &&
                             delta_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("delta shape debe coincidir con u: [",
                                            batch_size, ", ", seq_len, ", ", d_model, "]"));

        // A debe ser 1D: [d_model]
        OP_REQUIRES(context, A_tensor.dims() == 1,
                    errors::InvalidArgument("A requiere tensor 1D [d_model]. Recibido: ",
                                            A_tensor.dims(), "D"));
        OP_REQUIRES(context, A_tensor.dim_size(0) == d_model,
                    errors::InvalidArgument("A dim(0) debe ser d_model=", d_model,
                                            ". Recibido: ", A_tensor.dim_size(0)));

        // B debe ser 3D con misma shape que u
        OP_REQUIRES(context, B_tensor.dims() == 3,
                    errors::InvalidArgument("B requiere tensor 3D. Recibido: ",
                                            B_tensor.dims(), "D"));
        OP_REQUIRES(context, B_tensor.dim_size(0) == batch_size &&
                             B_tensor.dim_size(1) == seq_len &&
                             B_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("B shape debe coincidir con u: [",
                                            batch_size, ", ", seq_len, ", ", d_model, "]"));

        // C debe ser 3D con misma shape que u
        OP_REQUIRES(context, C_tensor.dims() == 3,
                    errors::InvalidArgument("C requiere tensor 3D. Recibido: ",
                                            C_tensor.dims(), "D"));
        OP_REQUIRES(context, C_tensor.dim_size(0) == batch_size &&
                             C_tensor.dim_size(1) == seq_len &&
                             C_tensor.dim_size(2) == d_model,
                    errors::InvalidArgument("C shape debe coincidir con u: [",
                                            batch_size, ", ", seq_len, ", ", d_model, "]"));

        // ====================================================================
        // PREPARAR TENSOR DE SALIDA
        // ====================================================================
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // ====================================================================
        // EXTRACCIÓN DE PUNTEROS VRAM
        // ====================================================================
        // Al ejecutarse bajo 'DEVICE_GPU', TensorFlow alojó la memoria implícitamente en la tarjeta.
        // '.data()' expone la dirección C subyacente real en VRAM. No hay cudaMemcpyHostToDevice
        // porque los datos YA RESIDEN en la tarjeta gráfica.
        const float* d_in_ptr    = input_tensor.flat<float>().data();
        float*       d_out_ptr   = output_tensor->flat<float>().data();
        const float* d_delta_ptr = delta_tensor.flat<float>().data();
        const float* d_A_ptr     = A_tensor.flat<float>().data();
        const float* d_B_ptr     = B_tensor.flat<float>().data();
        const float* d_C_ptr     = C_tensor.flat<float>().data();

        // ====================================================================
        // INVOCACIÓN AL DOMINIO NVIDIA/CUDA
        // ====================================================================
        int cuda_status = LaunchMambaSelectiveScan(
            d_in_ptr, d_out_ptr,
            d_delta_ptr, d_A_ptr, d_B_ptr, d_C_ptr,
            batch_size, seq_len, d_model);

        // Atrapamos la posible excepción de hardware asíncrono y se la empujamos a Python
        // para que Colab nos lo reporte inmediatamente en la celda.
        // '0' es cudaSuccess numérico.
        OP_REQUIRES(context, cuda_status == 0,
                    errors::Internal("GPU Ejecución falló en LaunchMambaSelectiveScan. Error CUDA: ", cuda_status));
    }
};

// 3. Registrar el Kernel para la GPU explícitamente.
// Esto indica al scheduler de TensorFlow que busque hardware gráfico acelerado para esto.
REGISTER_KERNEL_BUILDER(Name("MambaSelectiveScan").Device(DEVICE_GPU), MambaSelectiveScanOp);
