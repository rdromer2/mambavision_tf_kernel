"""
MambaVision TF Kernel — Fase 6: Bloque MambaVision Mixer
=========================================================
Capa Keras que ensambla el bloque completo del MambaVision Mixer:
    - Pre-normalización (LayerNorm)
    - Proyección dual de entrada (in_proj) con bifurcación x/z
    - Convolución depthwise 1D no-causal + SiLU (rama x)
    - Generación dinámica de parámetros SSM (dt, B, C)
    - Selective Scan CUDA custom (MambaSelectiveScanLayer)
    - Fusión multiplicativa por compuerta (gating con SiLU(z))
    - Proyección de salida + conexión residual

Uso:
    from mamba_block import MambaVisionMixerBlock

    block = MambaVisionMixerBlock(d_model=192, d_state=16)
    output = block(x)  # x: (batch, seq_len, d_model)
"""

import tensorflow as tf

# Importar la capa del Selective Scan validada en Fase 5.
# Al importar, se ejecuta automáticamente:
#   - Parche ctypes RTLD_GLOBAL
#   - Carga de libmamba_kernel.so / libmamba_bwd_kernel.so
#   - Registro de gradientes (con protección Jupyter)
from mamba_layer import MambaSelectiveScanLayer


class MambaVisionMixerBlock(tf.keras.layers.Layer):
    """
    Bloque completo del MambaVision Mixer.

    Implementa la topología de compuerta cruzada (gated architecture) donde
    la rama principal (SSM) se modula multiplicativamente con la rama de
    compuerta (z), seguido de una proyección de contracción y conexión
    residual.

    Args:
        d_model:          Dimensión del canal de entrada/salida.
        d_state:          Dimensión del espacio de estados oculto N (default 16).
        dt_rank:          Rango de la proyección intermedia de delta.
                          Default: d_model // 16.
        expand_ratio:     Factor de expansión interna de canales.
                          Default 1 (MambaVision). Mamba estándar usa 2.
        conv_kernel_size: Tamaño del kernel Conv1D depthwise (default 3).

    Inputs:
        inputs: Tensor (batch, seq_len, d_model).

    Output:
        Tensor (batch, seq_len, d_model) — misma forma que la entrada.

    Pesos entrenables:
        - norm:      γ, β de LayerNorm
        - in_proj:   W (d_model → 2·d_inner)
        - conv1d:    Kernel depthwise (k, 1, d_inner)
        - x_proj:    W (d_inner → dt_rank)
        - B_gen:     W (d_inner → d_state) — sin equivalente PyTorch
        - C_gen:     W (d_inner → d_state) — sin equivalente PyTorch
        - dt_proj:   W (dt_rank → d_inner) + bias
        - B_proj:    W (d_state → d_inner)
        - C_proj:    W (d_state → d_inner)
        - ssm_layer: A (d_inner,)
        - D:         (d_inner,) skip connection
        - out_proj:  W (d_inner → d_model)
    """

    def __init__(self, d_model, d_state=16, dt_rank=None, expand_ratio=1,
                 conv_kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(d_model // 16, 1)
        self.expand_ratio = expand_ratio
        self.conv_kernel_size = conv_kernel_size
        self.d_inner = d_model * expand_ratio

    def build(self, input_shape):
        """
        Instancia todas las sublayers con sus pesos.

        input_shape: TensorShape (batch, seq_len, d_model).
        """
        d = self.d_inner

        # --- Pre-normalización ---
        self.norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5)

        # --- Proyección dual de entrada ---
        # Expande d_model → 2·d_inner para bifurcar en ramas x y z.
        self.in_proj = tf.keras.layers.Dense(
            d * 2, use_bias=False, name='in_proj')

        # --- Convolución depthwise 1D (solo rama x) ---
        # Padding simétrico (no causal): cada canal ve contexto completo.
        # groups=d_inner → convolución separable en profundidad.
        self.conv1d = tf.keras.layers.Conv1D(
            filters=d,
            kernel_size=self.conv_kernel_size,
            padding='same',
            groups=d,
            name='conv1d_x')

        # --- Proyección de delta (dt) ---
        # Genera solo dt_raw. B_raw y C_raw se generan independientemente.
        self.x_proj = tf.keras.layers.Dense(
            self.dt_rank,
            use_bias=False,
            name='x_proj')

        # --- Generadores independientes de B y C ---
        # Generan B_raw y C_raw directamente desde x_act.
        # Sin equivalente en PyTorch; mantienen inicialización local.
        self.B_gen = tf.keras.layers.Dense(
            self.d_state, use_bias=False, name='B_gen')
        self.C_gen = tf.keras.layers.Dense(
            self.d_state, use_bias=False, name='C_gen')

        # --- Expansión de delta ---
        # dt_rank → d_inner, con bias (el bias codifica la escala base de Δ).
        self.dt_proj = tf.keras.layers.Dense(
            d, use_bias=True, name='dt_proj')

        # --- Adaptación geométrica B/C ---
        # El kernel CUDA asume B, C con la misma dimensión que u (d_inner).
        # Estas Dense proyectan d_state → d_inner para empalmar con el .so.
        self.B_proj = tf.keras.layers.Dense(
            d, use_bias=False, name='B_proj')
        self.C_proj = tf.keras.layers.Dense(
            d, use_bias=False, name='C_proj')

        # --- Selective Scan (kernel CUDA) ---
        self.ssm_layer = MambaSelectiveScanLayer(d_model=d)

        # --- Skip connection D ---
        # Vector escalar por canal: y += x · D
        self.D = self.add_weight(
            name='D',
            shape=(d,),
            initializer='ones',
            trainable=True)

        # --- Proyección de salida ---
        # Contrae d_inner → d_model para empatar con la conexión residual.
        self.out_proj = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name='out_proj')

        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass del bloque MambaVision Mixer.

        Args:
            inputs: Tensor (batch, seq_len, d_model).

        Returns:
            Tensor (batch, seq_len, d_model).

        Flujo:
            1. Pre-normalización LayerNorm.
            2. Proyección dual in_proj → bifurcación x / z.
            3. Rama principal: Conv1D + SiLU → generación SSM params → Selective Scan.
            4. Skip connection D.
            5. Fusión multiplicativa: out_ssm * SiLU(z).
            6. Proyección out_proj + conexión residual.
        """
        residual = inputs

        # 1. Pre-normalización
        x = self.norm(inputs)

        # 2. Proyección dual → espacio expandido
        xz = self.in_proj(x)                          # (B, L, 2·d_inner)

        # 3. Bifurcación: rama principal (x) y rama compuerta (z)
        x_branch, z_branch = tf.split(xz, 2, axis=-1)  # cada una (B, L, d_inner)

        # ===== RAMA PRINCIPAL (SSM) =====

        # 4. Convolución depthwise + SiLU
        x_act = tf.keras.activations.silu(
            self.conv1d(x_branch))                     # (B, L, d_inner)

        # 5. Generación dinámica de parámetros SSM
        dt_raw = self.x_proj(x_act)                    # (B, L, dt_rank)
        B_raw  = self.B_gen(x_act)                     # (B, L, d_state)
        C_raw  = self.C_gen(x_act)                     # (B, L, d_state)

        # 6. Expansión de delta + Softplus (estrictamente positivo)
        delta = tf.nn.softplus(self.dt_proj(dt_raw))   # (B, L, d_inner)

        # 7. Adaptación geométrica B/C: d_state → d_inner
        B = self.B_proj(B_raw)                         # (B, L, d_inner)
        C = self.C_proj(C_raw)                         # (B, L, d_inner)

        # 8. Selective Scan (kernel CUDA custom)
        out_ssm = self.ssm_layer([x_act, delta, B, C]) # (B, L, d_inner)

        # 9. Skip connection D
        out_ssm = out_ssm + x_act * self.D

        # ===== FUSIÓN DE COMPUERTA =====

        # 10. Multiplicación elemento a elemento con SiLU(z)
        gated_out = out_ssm * tf.nn.silu(z_branch)

        # ===== PROYECCIÓN + RESIDUAL =====

        # 11. Contracción dimensional y conexión residual
        return self.out_proj(gated_out) + residual

    def get_config(self):
        """Serialización para model.save() / model.load()."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'dt_rank': self.dt_rank,
            'expand_ratio': self.expand_ratio,
            'conv_kernel_size': self.conv_kernel_size,
        })
        return config
