"""
MambaVision TF Kernel — Fase 5: Abstracción en Keras
=====================================================
Capa nativa de TensorFlow/Keras que encapsula la operación Selective Scan
implementada en C++/CUDA. Consolida la carga de librerías dinámicas, el
registro de gradientes y la interfaz de pesos entrenables en un único módulo.

Uso:
    from mamba_layer import MambaSelectiveScanLayer

    layer = MambaSelectiveScanLayer(d_model=768)
    out = layer([u, delta, B, C])
"""

import os
import ctypes

# ==============================================================================
# 1. PARCHE CRÍTICO DE SÍMBOLOS
# ==============================================================================
# TensorFlow empaqueta sus símbolos C++ en una librería compartida interna.
# Sin este parche, dlopen() no puede resolver las referencias cruzadas entre
# nuestra .so custom y el runtime de TF, causando `undefined symbol` al cargar.
ctypes.CDLL(
    '/usr/local/lib/python3.12/dist-packages/tensorflow/libtensorflow_framework.so.2',
    ctypes.RTLD_GLOBAL
)

import tensorflow as tf
from tensorflow.python.framework import ops

# ==============================================================================
# 2. CARGA DE LIBRERÍAS DINÁMICAS (Forward + Backward)
# ==============================================================================
# Las rutas se parametrizan con la variable de entorno MAMBA_KERNEL_DIR.
# Por defecto apunta al directorio de build estándar de Colab.
_KERNEL_DIR = os.environ.get(
    'MAMBA_KERNEL_DIR',
    '/content/mambavision_tf_kernel/build'
)

_fwd_module = tf.load_op_library(
    os.path.join(_KERNEL_DIR, 'libmamba_kernel.so'))
_bwd_module = tf.load_op_library(
    os.path.join(_KERNEL_DIR, 'libmamba_bwd_kernel.so'))

# ==============================================================================
# 3. REGISTRO DE GRADIENTES CON PROTECCIÓN JUPYTER
# ==============================================================================
# @ops.RegisterGradient solo puede invocarse UNA VEZ por nombre de operación
# en el runtime de Python. En entornos interactivos (Jupyter/Colab), re-ejecutar
# la celda que importa este módulo intenta registrar la misma clave por segunda
# vez, lanzando KeyError. El try/except lo absorbe silenciosamente.
try:
    @ops.RegisterGradient("MambaSelectiveScan")
    def _mamba_selective_scan_grad(op, grad):
        """
        Función puente invocada automáticamente por tf.GradientTape.

        Args:
            op:   La operación forward original. op.inputs contiene los 5
                  tensores de entrada retenidos en VRAM por TF.
            grad: El gradiente de la pérdida respecto a la salida (dy).

        Returns:
            Lista de 5 gradientes [du, d_delta, d_A, d_B, d_C],
            uno por cada entrada del forward, en el mismo orden.
        """
        u, delta, A, B, C = op.inputs

        # Orden de argumentos: 5 tensores originales + dy al final.
        # Coincide con REGISTER_OP("MambaSelectiveScanGrad") en mamba_ssm_bwd_op.cc
        du, d_delta, d_A, d_B, d_C = _bwd_module.mamba_selective_scan_grad(
            u, delta, A, B, C, grad
        )

        return [du, d_delta, d_A, d_B, d_C]
except KeyError:
    # La clave "MambaSelectiveScan" ya existe en el registro de gradientes.
    # Esto sucede al recargar el módulo sin reiniciar el kernel de Python.
    pass

# ==============================================================================
# 4. CAPA KERAS: MambaSelectiveScanLayer
# ==============================================================================
class MambaSelectiveScanLayer(tf.keras.layers.Layer):
    """
    Capa Keras que ejecuta la operación Selective Scan de Mamba en GPU.

    Encapsula el kernel CUDA custom y gestiona el peso entrenable A
    (matriz de estado del sistema dinámico SSM).

    Args:
        d_model: Dimensión del modelo (número de canales). Define el tamaño
                 de la matriz de estado A.

    Inputs:
        inputs: Tupla de 4 tensores (u, delta, B, C).
            - u:     [batch, seq_len, d_model] — Entrada secuencial.
            - delta: [batch, seq_len, d_model] — Step size de discretización.
            - B:     [batch, seq_len, d_model] — Matriz de entrada (input-dependent).
            - C:     [batch, seq_len, d_model] — Matriz de salida (input-dependent).

    Output:
        Tensor [batch, seq_len, d_model] — Resultado del selective scan.

    Peso entrenable:
        A: [d_model] — Matriz de estado del SSM. Inicializada con valores
           negativos para garantizar estabilidad dinámica (Ā = exp(A·Δ) ∈ (0,1)).

    Ejemplo:
        layer = MambaSelectiveScanLayer(d_model=768)
        out = layer([u, delta, B, C])  # Forward pass en GPU
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        """
        Declara el peso entrenable A.

        Keras inyecta input_shape como una lista de TensorShapes (uno por cada
        tensor de la tupla de entrada). No se desempaqueta; se usa exclusivamente
        self.d_model para definir la geometría de A.

        La inicialización en el rango [-1.0, -0.1] garantiza que:
            A_bar = exp(A * delta) ∈ (0, 1)
        produciendo un factor de decaimiento estable para el estado oculto.
        """
        self.A = self.add_weight(
            name='A',
            shape=(self.d_model,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-1.0, maxval=-0.1),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Ejecuta el Selective Scan en GPU.

        Args:
            inputs: Tupla (u, delta, B, C) de tensores en VRAM.

        Ciclo de vida de los tensores:
            1. u, delta, B, C → tensores dinámicos del grafo de TF (VRAM).
            2. self.A → peso persistente gestionado por Keras (VRAM).
               Incluido automáticamente en model.trainable_variables.
            3. _fwd_module.mamba_selective_scan → recibe 5 punteros VRAM,
               ejecuta el kernel CUDA forward.
            4. En el backward, GradientTape intercepta dy, invoca
               _mamba_selective_scan_grad, y el kernel CUDA backward
               devuelve dA que Keras usa para actualizar self.A via
               el optimizador.
        """
        u, delta, B, C = inputs
        return _fwd_module.mamba_selective_scan(u, delta, self.A, B, C)

    def get_config(self):
        """Serialización para model.save() / model.load()."""
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config
