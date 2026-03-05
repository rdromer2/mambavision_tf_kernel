## 5.2 Abstracción Topológica (NCHW vs NHWC)

Un desafío silente pero crítico asociado a la interpolación entre *frameworks* de Deep Learning reside en la topología predeterminada de los tensores. Tradicionalmente, PyTorch prioriza el formato de memoria `channels_first` (`NCHW`: Muestra, Canales, Alto, Ancho) ya que optimizaciones heredadas como cuDNN fueron inicialmente concebidas de esa manera. En contraste directo, Keras y TensorFlow operan bajo la filosofía `channels_last` (`NHWC`: Muestra, Alto, Ancho, Canales).

### Eliminación de Transposiciones Redundantes

En las aproximaciones tempranas de migración, se intentó inyectar operaciones de transposición (`tf.transpose`) en las fronteras de los bloques Mamba para preservar la lógica aritmética originaria de PyTorch. Desafortunadamente, estas permutaciones de ejes representan un cuello de botella grave a nivel de ancho de banda de memoria de la GPU, provocando fragmentación y copias de VRAM no contiguas.

Para solventar este detrimento de rendimiento, se reescribió la arquitectura matemática subyacente del sistema MambaVision asimilando una topología *channels_last* nativa y continua a lo largo de toda la inferencia. Al eliminar por completo las operaciones de transposición y reformular los índices de acceso multidimensional, se maximizó el *stride* lineal secuencial en la GPU, reduciendo considerablemente los requerimientos térmicos y espaciales del *forward pass*.

### Matriz de Equivalencia de Tensores

Para clarificar la traslación dimensional entre los entornos, se estructuró la siguiente matriz de equivalencias de variables críticas:

| Variable Semántica     | Topología PyTorch (NCHW) | Topología Keras (NHWC) | Justificación Numérica                                                  |
| ---------------------- | -------------------------- | ------------------------ | ----------------------------------------------------------------------- |
| Entradas / Activaciones| `[B, C, H, W]`           | `[B, H, W, C]`         | Alineado automático con las convoluciones `tf.keras.layers.Conv2D`.      |
| Matrices Paramétricas B| `[B, d_state, H*W]`      | `[B, H*W, d_state]`    | Evita transposición física antes del producto escalar secuencial iterativo.|
| Tensor de Paso $\Delta$| `[B, C, H*W]`            | `[B, H*W, C]`          | Propagación temporal directa sobre el eje profundo alineado a los hilos.|

Al basar todo el kernel C++/CUDA programado durante el proyecto sobre estos preceptos formales de contigüidad orientada a `channels_last`, se logró abstraer con éxito la capa más crítica eliminando el sobrecoste matricial.

---
**Pendiente de revisión:**
- *Dibujar un esquema visual mostrando un tensor rojo `NCHW` transponiéndose frente a un flujo continuo verde `NHWC` para enfatizar gráficamente este punto ante el tribunal.*
