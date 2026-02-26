#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// Declaración de la función externa en CUDA que invoca al kernel de NVIDIA.
// Retorna cudaError_t (como un int) bajo el capó.
extern "C" int LaunchMambaSelectiveScan(const float* d_in, float* d_out, int batch_size, int seq_len, int d_model);

// 1. Registro de la interfaz hacia Python
REGISTER_OP("MambaSelectiveScan")
    .Input("u: float")
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
        // Capturar el tensor de entrada
        const Tensor& input_tensor = context->input(0);

        // Validación estricta dimensional
        OP_REQUIRES(context, input_tensor.dims() == 3,
                    errors::InvalidArgument("MambaVision requiere un tensor 3D. Recibido: ",
                                            input_tensor.dims(), "D"));

        // Extraer las dimensiones subyacentes
        const int batch_size = input_tensor.dim_size(0);
        const int seq_len = input_tensor.dim_size(1);
        const int d_model = input_tensor.dim_size(2);

        // Preparar el tensor de salida
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // --- EXTRACCIÓN DE PUNTEROS VRAM ---
        // Al ejecutarse bajo 'DEVICE_GPU', TensorFlow alojó la memoria implícitamente en la tarjeta.
        // '.data()' expone la dirección C real. No utilizamos "HostToDevice" aquí porque los datos YA ESTÁN.
        // Las abstracciones de alto nivel lo ocultan, pero éste es el puente de memoria esencial.
        const float* d_in_ptr = input_tensor.flat<float>().data();
        float* d_out_ptr = output_tensor->flat<float>().data();

        // --- INVOCACIÓN AL DOMINIO NVIDIA/CUDA ---
        // Llamada a "LaunchMambaSelectiveScan" que configurará bloques e hilos.
        int cuda_status = LaunchMambaSelectiveScan(d_in_ptr, d_out_ptr, batch_size, seq_len, d_model);
        
        // Atrapamos la posible excepción de hardware asíncrono y se la empujamos a Python 
        // para que Colab nos lo reporte inmediatamente en la celda y no haga crashear el kernel Jupyter.
        // '0' es cudaSuccess numérico.
        OP_REQUIRES(context, cuda_status == 0,
                    errors::Internal("GPU Ejecución falló en LaunchMambaSelectiveScan. Error de CUDA: ", cuda_status));
    }
};

// 3. Registrar el Kernel para la GPU explícitamente.
// Esto indica al scheduler de TensorFlow que busque hardware gráfico acelerado para esto.
REGISTER_KERNEL_BUILDER(Name("MambaSelectiveScan").Device(DEVICE_GPU), MambaSelectiveScanOp);
