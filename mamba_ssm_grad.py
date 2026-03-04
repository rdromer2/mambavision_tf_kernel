import tensorflow as tf

# ==============================================================================
# PUENTE PYTHON: REGISTRO DE GRADIENTES PARA EL KERNEL CUSTOM
# ==============================================================================
# Este archivo conecta el ecosistema de autodiferenciación de TensorFlow
# (tf.GradientTape) con nuestro binario CUDA escrito a medida.

# Cargar la librería dinámica compilada que contiene la operación Backward.
try:
    _bwd_module = tf.load_op_library('/content/mambavision_tf_kernel/build/libmamba_bwd_kernel.so')
except tf.errors.NotFoundError:
    print("⚠️  No se encontró libmamba_bwd_kernel.so. Asegúrate de haber compilado el proyecto.")

# Decorador crítico: Intercepta cualquier llamada a gradient() sobre nuestra op original.
@tf.RegisterGradient("MambaSelectiveScan")
def _mamba_selective_scan_grad(op, dy):
    """
    Función puente invocada automáticamente en el backward pass.
    
    Argumentos:
        op: La operación original del forward pass. Contiene los tensores
            de entrada retenidos en VRAM (op.inputs).
        dy: El gradiente de la pérdida final respecto a la salida (y) de
            nuestra operación en el forward.
            
    Retorno:
        Lista de gradientes correspondientes uno-a-uno a las entradas del forward.
        [du, d_delta, d_A_param, d_B_param, d_C_param]
    """
    # 1. Recuperar los tensores originales del forward pass (residen en VRAM)
    u       = op.inputs[0]
    delta   = op.inputs[1]
    a_param = op.inputs[2]
    b_param = op.inputs[3]
    c_param = op.inputs[4]
    
    # 2. Invocar nuestro kernel custom de C++/CUDA para computar derivadas.
    # El wrapper de C++ (mamba_ssm_bwd_op.cc) recibirá estos tensores,
    # validará que dy cuadre en dimensiones, asignará un Workspace temporal
    # en VRAM, y despachará el algoritmo de doble barrido al hardware NVIDIA.
    du, d_delta, d_a, d_b, d_c = _bwd_module.mamba_selective_scan_grad(
        dy=dy,
        u=u,
        delta=delta,
        a_param=a_param,
        b_param=b_param,
        c_param=c_param
    )
    
    # 3. Retornar los gradientes al ecosistema de autodiferenciación.
    # El orden debe coincidir exactamente con el orden de las entradas en REGISTER_OP.
    return [du, d_delta, d_a, d_b, d_c]
