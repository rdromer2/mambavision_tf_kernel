"""
MambaVision TF Kernel — Fase 8: Migración de Pesos PyTorch → Keras
===================================================================
Script de conversión que carga un checkpoint .pth de MambaVision (PyTorch),
aplica las transposiciones geométricas y transformaciones especiales necesarias,
e inyecta los pesos en el modelo Keras equivalente.

Transformaciones especiales:
    - x_proj: Recorte tensorial [:dt_rank, :] para aislar la proyección de dt.
    - A_log:  Promedio sobre d_state → -exp() para obtener A escalar.
    - QKV:    Partición de la fusión qkv en Q, K, V reshapeados a
              [dim, num_heads, head_dim] para MultiHeadAttention de Keras.

Uso:
    from convert_weights import load_and_inject_weights
    from mambavision_model import MambaVisionModel

    model = MambaVisionModel()
    model(tf.random.normal([1, 224, 224, 3]))  # Build
    stats = load_and_inject_weights(model, 'mambavision_tiny.pth')
"""

import numpy as np

# torch se importa de forma diferida (solo disponible en Colab).
# numpy es suficiente para las transformaciones de pesos.


# ==============================================================================
# 1. CARGA DEL CHECKPOINT
# ==============================================================================
def load_pytorch_checkpoint(pth_path):
    """
    Carga un checkpoint .pth de PyTorch y retorna el state_dict
    como un diccionario de numpy arrays.

    Args:
        pth_path: Ruta al archivo .pth.

    Returns:
        dict[str, np.ndarray]: state_dict con valores numpy.
    """
    import torch
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    # Algunos checkpoints envuelven el state_dict bajo llave 'state_dict'
    # o 'model'. Detectamos automáticamente.
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and any(
            k.startswith('stem.') or k.startswith('levels.') for k in checkpoint):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"No se encuentra state_dict en el checkpoint. "
            f"Llaves de nivel superior: {list(checkpoint.keys())[:10]}")

    return {k: v.cpu().numpy() for k, v in state_dict.items()}


# ==============================================================================
# 2. FUNCIONES DE TRANSPOSICIÓN Y TRANSFORMACIÓN
# ==============================================================================
def transpose_weight(weight_np, rule):
    """
    Aplica la transposición geométrica según el tipo de capa.

    Args:
        weight_np: np.ndarray del peso original de PyTorch.
        rule:      Tipo de transposición:
                   'conv2d' → (2, 3, 1, 0)
                   'conv1d' → (2, 1, 0)
                   'dense'  → (1, 0)
                   'vector' → copia directa
                   'skip'   → retorna None (peso sin equivalente)

    Returns:
        np.ndarray transpuesto, o None si rule == 'skip'.
    """
    if rule == 'conv2d':
        return np.transpose(weight_np, (2, 3, 1, 0))
    elif rule == 'conv1d':
        return np.transpose(weight_np, (2, 1, 0))
    elif rule == 'dense':
        return np.transpose(weight_np, (1, 0))
    elif rule == 'vector':
        return weight_np.copy()
    elif rule == 'skip':
        return None
    else:
        raise ValueError(f"Regla de transposición desconocida: '{rule}'")


def transform_A(A_log_np):
    """
    Convierte A_log [d_inner, d_state] de PyTorch al vector A [d_inner]
    de nuestro kernel CUDA escalar.

    Procedimiento:
        1. Promediar a lo largo de la dimensión del estado (axis=1).
        2. Aplicar la transformación: A = -exp(A_log_mean).

    El resultado es un vector negativo que garantiza estabilidad dinámica:
        A_bar = exp(A * delta) ∈ (0, 1) cuando delta > 0.

    Args:
        A_log_np: np.ndarray de shape [d_inner, d_state].

    Returns:
        np.ndarray de shape [d_inner].
    """
    return -np.exp(A_log_np.mean(axis=1))


def slice_x_proj(x_proj_weight, dt_rank):
    """
    Extrae la porción de dt del peso fusionado x_proj de PyTorch.

    En PyTorch, x_proj.weight tiene shape [dt_rank + 2*d_state, d_inner].
    Las primeras dt_rank filas corresponden a la proyección de delta (dt).
    Las filas restantes (B, C) no tienen equivalente en nuestro Keras.

    Args:
        x_proj_weight: np.ndarray de shape [dt_rank + 2*d_state, d_inner].
        dt_rank:       Número de filas a extraer (rango de dt).

    Returns:
        np.ndarray de shape [d_inner, dt_rank] (transpuesto para Keras Dense).
    """
    dt_slice = x_proj_weight[:dt_rank, :]  # [dt_rank, d_inner]
    return np.transpose(dt_slice, (1, 0))   # [d_inner, dt_rank]


def split_qkv_weights(qkv_weight, qkv_bias, dim, num_heads):
    """
    Parte el peso QKV fusionado de PyTorch en Q, K, V separados y los
    reshapea al formato multidimensional exigido por Keras MultiHeadAttention.

    PyTorch QKV fusionado:
        qkv_weight: [3*dim, dim]  (Linear: out_features × in_features)
        qkv_bias:   [3*dim]

    Keras MultiHeadAttention (EinsumDense internos):
        query_kernel:  [dim, num_heads, head_dim]
        key_kernel:    [dim, num_heads, head_dim]
        value_kernel:  [dim, num_heads, head_dim]
        query_bias:    [num_heads, head_dim]
        key_bias:      [num_heads, head_dim]
        value_bias:    [num_heads, head_dim]

    Args:
        qkv_weight: np.ndarray de shape [3*dim, dim].
        qkv_bias:   np.ndarray de shape [3*dim], o None si sin bias.
        dim:        Dimensión del modelo (canales).
        num_heads:  Número de cabezas de atención.

    Returns:
        dict con llaves 'q_kernel', 'k_kernel', 'v_kernel',
                        'q_bias', 'k_bias', 'v_bias'.
    """
    head_dim = dim // num_heads

    # Partir en 3 segmentos de [dim, dim] cada uno
    q_w, k_w, v_w = np.split(qkv_weight, 3, axis=0)  # cada [dim, dim]

    # Transponer (PyTorch Linear → Keras Dense) y reshapear
    # PyTorch: [out, in] → Transpose → [in, out] → Reshape → [in, heads, head_dim]
    q_kernel = np.transpose(q_w, (1, 0)).reshape(dim, num_heads, head_dim)
    k_kernel = np.transpose(k_w, (1, 0)).reshape(dim, num_heads, head_dim)
    v_kernel = np.transpose(v_w, (1, 0)).reshape(dim, num_heads, head_dim)

    result = {
        'q_kernel': q_kernel,
        'k_kernel': k_kernel,
        'v_kernel': v_kernel,
    }

    if qkv_bias is not None:
        q_b, k_b, v_b = np.split(qkv_bias, 3, axis=0)  # cada [dim]
        result['q_bias'] = q_b.reshape(num_heads, head_dim)
        result['k_bias'] = k_b.reshape(num_heads, head_dim)
        result['v_bias'] = v_b.reshape(num_heads, head_dim)

    return result


def split_attn_proj(proj_weight, proj_bias, dim, num_heads):
    """
    Convierte la proyección de salida de atención de PyTorch al formato
    de Keras MultiHeadAttention._output_dense.

    PyTorch:
        proj_weight: [dim, dim] (Linear)
        proj_bias:   [dim]

    Keras _output_dense (EinsumDense):
        kernel: [num_heads, head_dim, dim]
        bias:   [dim]

    Args:
        proj_weight: np.ndarray de shape [dim, dim].
        proj_bias:   np.ndarray de shape [dim], o None.
        dim:         Dimensión del modelo.
        num_heads:   Número de cabezas.

    Returns:
        dict con llaves 'proj_kernel' y 'proj_bias'.
    """
    head_dim = dim // num_heads

    # PyTorch Linear [out, in] → transponer a [in, out]
    # in = dim, out = dim, pero in se descompone en [num_heads, head_dim]
    # Resultado: [num_heads, head_dim, dim]
    proj_kernel = np.transpose(proj_weight, (1, 0))  # [dim_in, dim_out]
    proj_kernel = proj_kernel.reshape(num_heads, head_dim, dim)

    result = {'proj_kernel': proj_kernel}
    if proj_bias is not None:
        result['proj_bias'] = proj_bias.copy()

    return result


# ==============================================================================
# 3. CONSTRUCCIÓN DEL MAPA DE LLAVES
# ==============================================================================
def build_key_map(depths=(3, 3, 9, 3),
                  dims=(80, 160, 320, 640),
                  num_transformers=(0, 0, 4, 1),
                  num_heads=8,
                  d_state=16):
    """
    Genera el diccionario de mapeo entre llaves de PyTorch y rutas de
    variables de Keras, junto con la regla de transposición para cada una.

    Las llaves con tratamiento especial (x_proj, A_log, QKV) se marcan
    con reglas dedicadas que la función inject_weights() maneja aparte.

    Args:
        depths:           Bloques totales por etapa.
        dims:             Canales por etapa.
        num_transformers: Transformers por etapa.
        num_heads:        Cabezas de atención.
        d_state:          Dimensión del espacio de estados original.

    Returns:
        dict[str, tuple]: { pytorch_key: (keras_var_path, rule) }
            Reglas:  'conv2d', 'conv1d', 'dense', 'vector',
                     'A_transform', 'x_proj_slice',
                     'qkv_weight', 'qkv_bias',
                     'attn_proj_weight', 'attn_proj_bias',
                     'skip'.
    """
    key_map = {}

    # ── Stem ──────────────────────────────────────────────────────────────
    stem_map = {
        'stem.0.weight':       ('stem/stem_conv1/kernel:0',           'conv2d'),
        'stem.0.bias':         ('stem/stem_conv1/bias:0',             'vector'),
        'stem.1.weight':       ('stem/stem_bn1/gamma:0',              'vector'),
        'stem.1.bias':         ('stem/stem_bn1/beta:0',               'vector'),
        'stem.1.running_mean': ('stem/stem_bn1/moving_mean:0',        'vector'),
        'stem.1.running_var':  ('stem/stem_bn1/moving_variance:0',    'vector'),
        'stem.2.weight':       ('stem/stem_conv2/kernel:0',           'conv2d'),
        'stem.2.bias':         ('stem/stem_conv2/bias:0',             'vector'),
        'stem.3.weight':       ('stem/stem_bn2/gamma:0',              'vector'),
        'stem.3.bias':         ('stem/stem_bn2/beta:0',               'vector'),
        'stem.3.running_mean': ('stem/stem_bn2/moving_mean:0',        'vector'),
        'stem.3.running_var':  ('stem/stem_bn2/moving_variance:0',    'vector'),
    }
    key_map.update(stem_map)

    # ── Clasificador (head) ──────────────────────────────────────────────
    head_map = {
        'head.weight': ('classifier/kernel:0', 'dense'),
        'head.bias':   ('classifier/bias:0',   'vector'),
        'norm.weight': ('final_norm/gamma:0',  'vector'),
        'norm.bias':   ('final_norm/beta:0',   'vector'),
    }
    key_map.update(head_map)

    # ── Downsamplers (transiciones entre etapas 1→2, 2→3, 3→4) ──────────
    for i in range(1, 4):
        prefix_pt = f'levels.{i}.downsample'
        prefix_kr = f'downsample_{i}'
        key_map.update({
            f'{prefix_pt}.conv.weight':       (f'{prefix_kr}/ds_conv/kernel:0',           'conv2d'),
            f'{prefix_pt}.conv.bias':         (f'{prefix_kr}/ds_conv/bias:0',             'vector'),
            f'{prefix_pt}.norm.weight':       (f'{prefix_kr}/ds_bn/gamma:0',              'vector'),
            f'{prefix_pt}.norm.bias':         (f'{prefix_kr}/ds_bn/beta:0',               'vector'),
            f'{prefix_pt}.norm.running_mean': (f'{prefix_kr}/ds_bn/moving_mean:0',        'vector'),
            f'{prefix_pt}.norm.running_var':  (f'{prefix_kr}/ds_bn/moving_variance:0',    'vector'),
        })

    # ── Bloques por etapa ────────────────────────────────────────────────
    for stage_idx in range(4):
        n_tf = num_transformers[stage_idx]
        n_mamba = depths[stage_idx] - n_tf
        dim = dims[stage_idx]

        # --- Bloques Mamba ---
        for j in range(n_mamba):
            pt = f'levels.{stage_idx}.blocks.{j}.mixer'
            kr = f'stage{stage_idx}_mamba{j}'

            key_map.update({
                # Proyección de entrada (sin bias)
                f'{pt}.in_proj.weight':
                    (f'{kr}/in_proj/kernel:0', 'dense'),

                # Conv1D depthwise
                f'{pt}.conv1d.weight':
                    (f'{kr}/conv1d_x/kernel:0', 'conv1d'),
                f'{pt}.conv1d.bias':
                    (f'{kr}/conv1d_x/bias:0', 'vector'),

                # x_proj: solo porción dt (recorte tensorial)
                f'{pt}.x_proj.weight':
                    (f'{kr}/x_proj/kernel:0', 'x_proj_slice'),

                # dt_proj
                f'{pt}.dt_proj.weight':
                    (f'{kr}/dt_proj/kernel:0', 'dense'),
                f'{pt}.dt_proj.bias':
                    (f'{kr}/dt_proj/bias:0', 'vector'),

                # A_log → transformación especial
                f'{pt}.A_log':
                    (f'{kr}/mamba_selective_scan_layer/A:0', 'A_transform'),

                # D (skip connection vector)
                f'{pt}.D':
                    (f'{kr}/D:0', 'vector'),

                # Proyección de salida (sin bias)
                f'{pt}.out_proj.weight':
                    (f'{kr}/out_proj/kernel:0', 'dense'),

                # LayerNorm pre-bloque
                f'{pt}.norm.weight':
                    (f'{kr}/layer_normalization/gamma:0', 'vector'),
                f'{pt}.norm.bias':
                    (f'{kr}/layer_normalization/beta:0', 'vector'),
            })

        # --- Bloques Transformer ---
        for j_tf in range(n_tf):
            # Índice absoluto en PyTorch (Mamba primero, luego Transformer)
            j_abs = n_mamba + j_tf
            pt = f'levels.{stage_idx}.blocks.{j_abs}'
            kr = f'stage{stage_idx}_transformer{j_tf}'

            key_map.update({
                # Normas
                f'{pt}.norm1.weight':
                    (f'{kr}/attn_norm/gamma:0', 'vector'),
                f'{pt}.norm1.bias':
                    (f'{kr}/attn_norm/beta:0', 'vector'),
                f'{pt}.norm2.weight':
                    (f'{kr}/mlp_norm/gamma:0', 'vector'),
                f'{pt}.norm2.bias':
                    (f'{kr}/mlp_norm/beta:0', 'vector'),

                # QKV fusionado → marcado para tratamiento especial
                f'{pt}.attn.qkv.weight':
                    (f'{kr}/mhsa', 'qkv_weight'),
                f'{pt}.attn.qkv.bias':
                    (f'{kr}/mhsa', 'qkv_bias'),

                # Proyección de salida de atención
                f'{pt}.attn.proj.weight':
                    (f'{kr}/mhsa', 'attn_proj_weight'),
                f'{pt}.attn.proj.bias':
                    (f'{kr}/mhsa', 'attn_proj_bias'),

                # MLP
                f'{pt}.mlp.fc1.weight':
                    (f'{kr}/mlp_fc1/kernel:0', 'dense'),
                f'{pt}.mlp.fc1.bias':
                    (f'{kr}/mlp_fc1/bias:0', 'vector'),
                f'{pt}.mlp.fc2.weight':
                    (f'{kr}/mlp_fc2/kernel:0', 'dense'),
                f'{pt}.mlp.fc2.bias':
                    (f'{kr}/mlp_fc2/bias:0', 'vector'),
            })

    return key_map


# ==============================================================================
# 4. INYECCIÓN DE PESOS
# ==============================================================================
def _find_variable(model, var_path):
    """
    Localiza una variable de Keras por su ruta (e.g., 'stem/stem_conv1/kernel:0').

    Recorre model.variables buscando coincidencia parcial en el nombre.
    Keras prefija los nombres de las variables con el nombre del modelo,
    así que buscamos un sufijo que contenga la ruta.

    Args:
        model:    tf.keras.Model construido.
        var_path: Ruta parcial de la variable (sin prefijo de modelo).

    Returns:
        tf.Variable, o None si no se encuentra.
    """
    # Eliminar el ':0' final para la búsqueda
    search_path = var_path.rstrip(':0').rstrip(':')

    for var in model.variables:
        # var.name es algo como 'mamba_vision_model/stem/stem_conv1/kernel:0'
        # Buscamos si termina con nuestra ruta (sin el prefijo del modelo)
        var_name_clean = var.name.split(':')[0]  # quitar :0
        if var_name_clean.endswith(search_path):
            return var

    return None


def _find_mha_layer(model, mha_path):
    """
    Localiza una capa MultiHeadAttention en el modelo por su ruta.

    Args:
        model:    tf.keras.Model.
        mha_path: Ruta parcial, e.g. 'stage2_transformer0/mhsa'.

    Returns:
        tf.keras.layers.MultiHeadAttention, o None.
    """
    # Extraer nombre del bloque y del MHA
    parts = mha_path.split('/')
    block_name = parts[0]   # e.g., 'stage2_transformer0'
    mha_name = parts[1]     # e.g., 'mhsa'

    # Buscar el bloque Transformer en el modelo
    for stage_transformers in model.transformer_blocks:
        for block in stage_transformers:
            if block.name == block_name:
                # El MHA es un atributo del bloque
                mha = getattr(block, 'mhsa', None)
                if mha is not None and mha.name == mha_name:
                    return mha

    return None


def inject_weights(model, state_dict, key_map,
                   depths=(3, 3, 9, 3),
                   dims=(80, 160, 320, 640),
                   num_transformers=(0, 0, 4, 1),
                   num_heads=8,
                   d_state=16,
                   verbose=True):
    """
    Itera el mapa de llaves, aplica transposiciones/transformaciones y
    asigna cada peso al modelo Keras.

    Args:
        model:            tf.keras.Model construido (con .build() ejecutado).
        state_dict:       dict[str, np.ndarray] del checkpoint de PyTorch.
        key_map:          dict generado por build_key_map().
        depths, dims, etc.: Parámetros arquitectónicos.
        num_heads:        Cabezas de atención.
        d_state:          Dimensión del estado original de PyTorch.
        verbose:          Si True, imprime progreso.

    Returns:
        dict con estadísticas: 'injected', 'skipped', 'missing', 'errors'.
    """
    import tensorflow as tf

    stats = {
        'injected': [],
        'skipped': [],
        'missing_in_pytorch': [],
        'missing_in_keras': [],
        'errors': [],
    }

    # Calcular dt_rank para cada etapa (d_inner // 16)
    # Con expand_ratio=1, d_inner = dim
    dt_ranks = {i: max(dims[i] // 16, 1) for i in range(4)}

    # Buffer para acumular pesos QKV por bloque MHA antes de inyectar
    # Estructura: { mha_path: { 'qkv_weight': ..., 'qkv_bias': ...,
    #                           'proj_weight': ..., 'proj_bias': ... } }
    mha_buffer = {}

    for pt_key, (kr_path, rule) in key_map.items():
        # Verificar que la llave existe en el state_dict de PyTorch
        if pt_key not in state_dict:
            stats['missing_in_pytorch'].append(pt_key)
            if verbose:
                print(f"  [SKIP] PyTorch key no encontrada: {pt_key}")
            continue

        weight_np = state_dict[pt_key]

        # ── Reglas simples (transposición directa) ────────────────────
        if rule in ('conv2d', 'conv1d', 'dense', 'vector'):
            transformed = transpose_weight(weight_np, rule)
            var = _find_variable(model, kr_path)
            if var is None:
                stats['missing_in_keras'].append(kr_path)
                if verbose:
                    print(f"  [MISS] Variable Keras no encontrada: {kr_path}")
                continue

            try:
                var.assign(transformed)
                stats['injected'].append(pt_key)
                if verbose:
                    print(f"  [OK]   {pt_key} → {kr_path}  "
                          f"({weight_np.shape} → {transformed.shape})")
            except Exception as e:
                stats['errors'].append((pt_key, str(e)))
                if verbose:
                    print(f"  [ERR]  {pt_key}: {e}")

        # ── x_proj: Recorte tensorial de dt ───────────────────────────
        elif rule == 'x_proj_slice':
            # Determinar la etapa a partir de la llave de PyTorch
            # e.g., 'levels.2.blocks.0.mixer.x_proj.weight'
            stage_idx = int(pt_key.split('.')[1])
            dt_rank = dt_ranks[stage_idx]

            transformed = slice_x_proj(weight_np, dt_rank)
            var = _find_variable(model, kr_path)
            if var is None:
                stats['missing_in_keras'].append(kr_path)
                if verbose:
                    print(f"  [MISS] Variable Keras no encontrada: {kr_path}")
                continue

            try:
                var.assign(transformed)
                stats['injected'].append(pt_key)
                if verbose:
                    print(f"  [OK]   {pt_key} → {kr_path}  "
                          f"(slice [{dt_rank}, :] → {transformed.shape})")
            except Exception as e:
                stats['errors'].append((pt_key, str(e)))
                if verbose:
                    print(f"  [ERR]  {pt_key}: {e}")

        # ── A_log: Transformación especial ────────────────────────────
        elif rule == 'A_transform':
            transformed = transform_A(weight_np)
            var = _find_variable(model, kr_path)
            if var is None:
                stats['missing_in_keras'].append(kr_path)
                if verbose:
                    print(f"  [MISS] Variable Keras no encontrada: {kr_path}")
                continue

            try:
                var.assign(transformed)
                stats['injected'].append(pt_key)
                if verbose:
                    print(f"  [OK]   {pt_key} → {kr_path}  "
                          f"(A_log {weight_np.shape} → A {transformed.shape})")
            except Exception as e:
                stats['errors'].append((pt_key, str(e)))
                if verbose:
                    print(f"  [ERR]  {pt_key}: {e}")

        # ── QKV / Attn Proj: Acumular en buffer ──────────────────────
        elif rule in ('qkv_weight', 'qkv_bias',
                      'attn_proj_weight', 'attn_proj_bias'):
            if kr_path not in mha_buffer:
                mha_buffer[kr_path] = {}
            mha_buffer[kr_path][rule] = weight_np

        elif rule == 'skip':
            stats['skipped'].append(pt_key)
            if verbose:
                print(f"  [SKIP] {pt_key} (sin equivalente en Keras)")

    # ── Inyectar pesos QKV acumulados en los MHA de Keras ─────────────
    for mha_path, buffers in mha_buffer.items():
        mha_layer = _find_mha_layer(model, mha_path)
        if mha_layer is None:
            stats['missing_in_keras'].append(mha_path)
            if verbose:
                print(f"  [MISS] MHA layer no encontrada: {mha_path}")
            continue

        # Inferir dim y num_heads de la capa
        # El mha_path contiene el nombre del stage, e.g. 'stage2_transformer0/mhsa'
        block_name = mha_path.split('/')[0]
        # Extraer stage_idx del nombre
        stage_idx = int(block_name.split('_')[0].replace('stage', ''))
        dim = dims[stage_idx]

        try:
            # ─ QKV ─
            if 'qkv_weight' in buffers:
                qkv_w = buffers['qkv_weight']
                qkv_b = buffers.get('qkv_bias', None)
                qkv = split_qkv_weights(qkv_w, qkv_b, dim, num_heads)

                # Inyectar en las EinsumDense internas del MHA
                mha_layer._query_dense.kernel.assign(qkv['q_kernel'])
                mha_layer._key_dense.kernel.assign(qkv['k_kernel'])
                mha_layer._value_dense.kernel.assign(qkv['v_kernel'])

                if 'q_bias' in qkv:
                    mha_layer._query_dense.bias.assign(qkv['q_bias'])
                    mha_layer._key_dense.bias.assign(qkv['k_bias'])
                    mha_layer._value_dense.bias.assign(qkv['v_bias'])

                stats['injected'].append(f'{mha_path}/qkv')
                if verbose:
                    print(f"  [OK]   QKV → {mha_path}  "
                          f"({qkv_w.shape} → Q/K/V "
                          f"[{dim}, {num_heads}, {dim // num_heads}])")

            # ─ Proyección de salida ─
            if 'attn_proj_weight' in buffers:
                proj_w = buffers['attn_proj_weight']
                proj_b = buffers.get('attn_proj_bias', None)
                proj = split_attn_proj(proj_w, proj_b, dim, num_heads)

                mha_layer._output_dense.kernel.assign(proj['proj_kernel'])
                if 'proj_bias' in proj:
                    mha_layer._output_dense.bias.assign(proj['proj_bias'])

                stats['injected'].append(f'{mha_path}/proj')
                if verbose:
                    print(f"  [OK]   Proj → {mha_path}  "
                          f"({proj_w.shape} → "
                          f"[{num_heads}, {dim // num_heads}, {dim}])")

        except Exception as e:
            stats['errors'].append((mha_path, str(e)))
            if verbose:
                print(f"  [ERR]  MHA {mha_path}: {e}")

    return stats


# ==============================================================================
# 5. FUNCIÓN PRINCIPAL DE ALTO NIVEL
# ==============================================================================
def load_and_inject_weights(model, pth_path,
                            depths=(3, 3, 9, 3),
                            dims=(80, 160, 320, 640),
                            num_transformers=(0, 0, 4, 1),
                            num_heads=8,
                            d_state=16,
                            verbose=True):
    """
    Función principal: carga un checkpoint .pth de PyTorch y migra todos
    los pesos compatibles al modelo Keras de MambaVision.

    Esta función se puede importar y llamar directamente desde una celda
    de Jupyter/Colab:

        from convert_weights import load_and_inject_weights
        stats = load_and_inject_weights(model, 'mambavision_tiny.pth')

    Args:
        model:            MambaVisionModel de Keras, ya construido.
        pth_path:         Ruta al archivo .pth de PyTorch.
        depths:           Bloques totales por etapa.
        dims:             Canales por etapa.
        num_transformers: Transformers por etapa.
        num_heads:        Cabezas de atención.
        d_state:          Dimensión del espacio de estados original.
        verbose:          Imprimir progreso detallado.

    Returns:
        dict con estadísticas de la inyección:
            'injected':           llaves inyectadas exitosamente.
            'skipped':            llaves marcadas como skip.
            'missing_in_pytorch': llaves esperadas pero ausentes en el .pth.
            'missing_in_keras':   variables Keras no encontradas.
            'errors':             errores de asignación (shape mismatch, etc.).
    """
    if verbose:
        print("=" * 60)
        print("Fase 8: Migración de Pesos PyTorch → Keras")
        print("=" * 60)

    # 1. Cargar checkpoint
    if verbose:
        print(f"\n[1/3] Cargando checkpoint: {pth_path}")
    state_dict = load_pytorch_checkpoint(pth_path)
    if verbose:
        print(f"      {len(state_dict)} llaves encontradas en state_dict.")

    # 2. Construir mapa de llaves
    if verbose:
        print(f"\n[2/3] Construyendo mapa de llaves...")
    key_map = build_key_map(depths, dims, num_transformers, num_heads, d_state)
    if verbose:
        print(f"      {len(key_map)} mapeos definidos.")

    # 3. Inyectar pesos
    if verbose:
        print(f"\n[3/3] Inyectando pesos:\n")
    stats = inject_weights(
        model, state_dict, key_map,
        depths=depths, dims=dims,
        num_transformers=num_transformers,
        num_heads=num_heads, d_state=d_state,
        verbose=verbose)

    # Resumen
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RESUMEN DE MIGRACIÓN:")
        print(f"  Inyectados:             {len(stats['injected'])}")
        print(f"  Omitidos (skip):        {len(stats['skipped'])}")
        print(f"  Ausentes en PyTorch:    {len(stats['missing_in_pytorch'])}")
        print(f"  No encontrados en Keras:{len(stats['missing_in_keras'])}")
        print(f"  Errores:                {len(stats['errors'])}")
        print(f"{'=' * 60}")

        # Listar llaves del state_dict que NO se mapearon
        mapped_keys = set(key_map.keys())
        unmapped = set(state_dict.keys()) - mapped_keys
        if unmapped:
            print(f"\n  Llaves de PyTorch sin mapear ({len(unmapped)}):")
            for k in sorted(unmapped):
                print(f"    - {k}  {state_dict[k].shape}")

    return stats


# ==============================================================================
# 6. VALIDACIÓN NUMÉRICA
# ==============================================================================
def validate_output(model, seed=0):
    """
    Ejecuta inferencia con input determinista y valida las propiedades
    del output (shape, NaN, Inf, rango de valores).

    Args:
        model: MambaVisionModel de Keras con pesos inyectados.
        seed:  Semilla para el input determinista.

    Returns:
        dict con resultados: 'shape', 'has_nan', 'has_inf',
                             'min', 'max', 'output'.
    """
    import tensorflow as tf

    np.random.seed(seed)
    img = np.random.randn(1, 224, 224, 3).astype(np.float32)

    out = model(tf.constant(img), training=False).numpy()

    results = {
        'shape': out.shape,
        'has_nan': bool(np.any(np.isnan(out))),
        'has_inf': bool(np.any(np.isinf(out))),
        'min': float(out.min()),
        'max': float(out.max()),
        'output': out,
    }

    print(f"\n{'=' * 40}")
    print(f"VALIDACIÓN NUMÉRICA:")
    print(f"  Shape:    {results['shape']}")
    print(f"  Has NaN:  {results['has_nan']}")
    print(f"  Has Inf:  {results['has_inf']}")
    print(f"  Rango:    [{results['min']:.4f}, {results['max']:.4f}]")
    print(f"{'=' * 40}")

    assert results['shape'] == (1, 1000), \
        f"Shape mismatch: esperado (1, 1000), obtenido {results['shape']}"
    assert not results['has_nan'], "Output contiene NaN"
    assert not results['has_inf'], "Output contiene Inf"

    return results
