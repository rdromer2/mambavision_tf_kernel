"""
MambaVision TF Kernel — Fase 7: Macroarquitectura MambaVision
==============================================================
Modelo jerárquico de 4 etapas que combina bloques MambaVisionMixer
y bloques Transformer de autoatención para clasificación de imágenes.

Clases:
    - PatchEmbedding:     Stem con 2× Conv2D stride 2.
    - Downsample:         Reducción espacial ×2 entre etapas.
    - TransformerBlock:   MHSA + MLP (pre-norm) con expansión ×4.
    - MambaVisionModel:   Orquestador de 4 etapas + cabeza de clasificación.

Uso:
    from mambavision_model import MambaVisionModel

    model = MambaVisionModel()  # Variante Tiny por defecto
    logits = model(tf.random.normal([1, 224, 224, 3]), training=False)
"""

import tensorflow as tf
from mamba_block import MambaVisionMixerBlock


# ==============================================================================
# 1. PATCH EMBEDDING (Stem)
# ==============================================================================
class PatchEmbedding(tf.keras.layers.Layer):
    """
    Convierte imagen raw en mapa de características inicial.

    2 capas Conv2D con stride 2 cada una → reducción espacial total ×4.
    Para una imagen 224×224, la salida es 56×56.

    Args:
        dim: Canales de salida del stem (dims[0] de la arquitectura).
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            self.dim // 2, kernel_size=3, strides=2,
            padding='same', name='stem_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='stem_bn1')

        self.conv2 = tf.keras.layers.Conv2D(
            self.dim, kernel_size=3, strides=2,
            padding='same', name='stem_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='stem_bn2')

        super().build(input_shape)

    def call(self, x, training=False):
        """
        Args:
            x: (B, H, W, 3)
        Returns:
            (B, H/4, W/4, dim)
        """
        x = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.gelu(self.bn2(self.conv2(x), training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dim': self.dim})
        return config


# ==============================================================================
# 2. DOWNSAMPLE
# ==============================================================================
class Downsample(tf.keras.layers.Layer):
    """
    Reduce resolución espacial ×2 y expande canales entre etapas.

    Args:
        dim_out: Canales de salida (dims[i+1] de la arquitectura).
    """

    def __init__(self, dim_out, **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            self.dim_out, kernel_size=3, strides=2,
            padding='same', name='ds_conv')
        self.bn = tf.keras.layers.BatchNormalization(name='ds_bn')
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Args:
            x: (B, H, W, C_in)
        Returns:
            (B, H/2, W/2, dim_out)
        """
        return self.bn(self.conv(x), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({'dim_out': self.dim_out})
        return config


# ==============================================================================
# 3. TRANSFORMER BLOCK
# ==============================================================================
class TransformerBlock(tf.keras.layers.Layer):
    """
    Bloque estándar de autoatención con pre-normalización.

    Estructura:
        Sub-bloque 1: LayerNorm → MHSA → residual
        Sub-bloque 2: LayerNorm → MLP(×4) → residual

    Args:
        dim:       Dimensión del canal.
        num_heads: Número de cabezas de atención.
    """

    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads

    def build(self, input_shape):
        # --- Sub-bloque atención ---
        self.norm1 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name='attn_norm')
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            name='mhsa')

        # --- Sub-bloque MLP (expansión ×4) ---
        self.norm2 = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name='mlp_norm')
        self.fc1 = tf.keras.layers.Dense(
            self.dim * 4, name='mlp_fc1')
        self.fc2 = tf.keras.layers.Dense(
            self.dim, name='mlp_fc2')

        super().build(input_shape)

    def call(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # --- Atención ---
        residual = x
        x_norm = self.norm1(x)
        x = self.mhsa(query=x_norm, value=x_norm, key=x_norm) + residual

        # --- MLP (expansión ×4) ---
        residual = x
        h = self.norm2(x)
        h = tf.nn.gelu(self.fc1(h))       # (B, L, dim * 4)
        h = self.fc2(h)                    # (B, L, dim)
        return h + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
        })
        return config


# ==============================================================================
# 4. MAMBAVISION MODEL
# ==============================================================================
class MambaVisionModel(tf.keras.Model):
    """
    Modelo MambaVision completo: backbone híbrido Mamba-Transformer
    con cabeza de clasificación.

    Variante Tiny (T) por defecto:
        depths          = [3, 3, 9, 3]
        dims            = [80, 160, 320, 640]
        num_transformers = [0, 0, 4, 1]

    Args:
        depths:           Bloques totales por etapa.
        dims:             Canales por etapa.
        num_transformers: Transformers por etapa.
                          Mamba por etapa = depths[i] - num_transformers[i].
        num_heads:        Cabezas de atención (escalar, todos los Transformer).
        num_classes:      Clases de salida.
        d_state:          Dimensión del espacio de estados del SSM.
    """

    def __init__(self,
                 depths=(3, 3, 9, 3),
                 dims=(80, 160, 320, 640),
                 num_transformers=(0, 0, 4, 1),
                 num_heads=8,
                 num_classes=1000,
                 d_state=16,
                 **kwargs):
        super().__init__(**kwargs)
        self.depths = list(depths)
        self.dims = list(dims)
        self.num_transformers = list(num_transformers)
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.d_state = d_state

        # --- Stem ---
        self.stem = PatchEmbedding(dims[0], name='stem')

        # --- Downsamplers (3 transiciones entre 4 etapas) ---
        self.downsamplers = []
        for i in range(1, 4):
            self.downsamplers.append(
                Downsample(dims[i], name=f'downsample_{i}'))

        # --- Bloques por etapa ---
        self.mamba_blocks = []
        self.transformer_blocks = []

        for i in range(4):
            n_transformer = num_transformers[i]
            n_mamba = depths[i] - n_transformer

            # Bloques Mamba (primera porción de la etapa)
            stage_mamba = []
            for j in range(n_mamba):
                stage_mamba.append(
                    MambaVisionMixerBlock(
                        d_model=dims[i],
                        d_state=d_state,
                        name=f'stage{i}_mamba{j}'))
            self.mamba_blocks.append(stage_mamba)

            # Bloques Transformer (segunda porción, solo etapas 3-4)
            stage_transformer = []
            for j in range(n_transformer):
                stage_transformer.append(
                    TransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads,
                        name=f'stage{i}_transformer{j}'))
            self.transformer_blocks.append(stage_transformer)

        # --- Cabeza de clasificación ---
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.final_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name='final_norm')
        self.head = tf.keras.layers.Dense(
            num_classes, name='classifier')

    def call(self, x, training=False):
        """
        Forward pass completo: imagen → logits.

        Args:
            x: (B, 224, 224, 3) — imagen de entrada.
            training: bool — activa estadísticas en línea de BatchNorm.

        Returns:
            (B, num_classes) — logits de clasificación.
        """
        # Stem: (B, 224, 224, 3) → (B, 56, 56, dims[0])
        x = self.stem(x, training=training)

        for i in range(4):
            # Downsample entre etapas (no en la primera)
            if i > 0:
                x = self.downsamplers[i - 1](x, training=training)

            # Guardar tensor 2D ANTES de aplanar para reconstruir después
            x_2d = x
            B = tf.shape(x)[0]
            C = x.shape[-1]               # Dimensión estática (canales)

            # Aplanar 2D → 1D: (B, H, W, C) → (B, H·W, C)
            x = tf.reshape(x, [B, -1, C])

            # Bloques Mamba (primera mitad de la etapa)
            for block in self.mamba_blocks[i]:
                x = block(x)

            # Bloques Transformer (segunda mitad, solo etapas 3-4)
            for block in self.transformer_blocks[i]:
                x = block(x)

            # Reconstruir 2D: (B, H·W, C) → (B, H, W, C)
            H = tf.shape(x_2d)[1]
            W = tf.shape(x_2d)[2]
            x = tf.reshape(x, [B, H, W, C])

        # Clasificación: (B, H, W, C) → (B, C) → (B, num_classes)
        x = self.pool(x)                  # (B, dims[-1])
        x = self.final_norm(x)            # (B, dims[-1])
        x = self.head(x)                  # (B, num_classes)
        return x
