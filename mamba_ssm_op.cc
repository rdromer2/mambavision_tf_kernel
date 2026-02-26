#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// 1. Registro de la interfaz hacia Python
// Definimos que entra un tensor 'u' (float) y sale un tensor 'out' (float)
REGISTER_OP("MambaSelectiveScan")
    .Input("u: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Le decimos a TensorFlow que la forma de salida es idéntica a la de entrada
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    });

// 2. La clase del Kernel (El Centinela)
class MambaSelectiveScanOp : public OpKernel {
public:
    explicit MambaSelectiveScanOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Capturar el tensor de entrada
        const Tensor& input_tensor = context->input(0);

        // --- INICIO DE VALIDACIONES ESTRICTAS ---
        // El detalle crítico: OP_REQUIRES detiene la ejecución y devuelve el error a Python
        // si la condición falla, evitando un Segmentation Fault en C++.

        // Validación A: Asegurar que el tensor tiene exactamente 3 dimensiones (Batch, Seq, Dim)
        OP_REQUIRES(context, input_tensor.dims() == 3,
                    errors::InvalidArgument("MambaVision requiere un tensor 3D. Recibido: ",
                                            input_tensor.dims(), "D"));

        // Extraer las dimensiones para uso futuro en CUDA
        const int batch_size = input_tensor.dim_size(0);
        const int seq_len = input_tensor.dim_size(1);
        const int d_model = input_tensor.dim_size(2);

        // Preparar el tensor de salida con la misma forma
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // --- EXTRACCIÓN DE PUNTEROS CRUDOS ---
        // flat<float>() aplana la vista del tensor N-dimensional a 1D.
        // .data() extrae la dirección de memoria física RAM/VRAM del primer número.
        const float* input_ptr = input_tensor.flat<float>().data();
        float* output_ptr = output_tensor->flat<float>().data();

        // Operación temporal (Dummy) para validar el puente antes de integrar CUDA en la Fase 3.
        // Simplemente copiamos la memoria de entrada a la de salida.
        const int total_elements = batch_size * seq_len * d_model;
        for (int i = 0; i < total_elements; ++i) {
            output_ptr[i] = input_ptr[i]; 
        }
    }
};

// 3. Registrar el Kernel en el sistema para CPU (más adelante lo registraremos para GPU)
REGISTER_KERNEL_BUILDER(Name("MambaSelectiveScan").Device(DEVICE_CPU), MambaSelectiveScanOp);
