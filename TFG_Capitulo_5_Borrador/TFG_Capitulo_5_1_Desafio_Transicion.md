# Capítulo 5: Migración de MambaVision a Keras

## 5.1 El Desafío de la Transición

La adopción de la arquitectura MambaVision para tareas de clasificación dermatológica requiere lidiar con un entorno de desarrollo eminentemente optimizado para la investigación académica rápida, comúnmente soportado en PyTorch y el lenguaje Triton. Sin embargo, para la fase de producción, integración clínica y validación en despliegues estandarizados, se identificó la necesidad imperativa de migrar el sistema hacia el ecosistema de Keras y TensorFlow. 

La implementación original del componente central, el mecanismo *Selective Scan*, depende del compilador Triton para la paralelización en GPU y de directivas estrictas de PyTorch que no poseen equivalentes directos asincrónicos en Keras. Esta disparidad arquitectónica impidió una transcripción directa del código y obligó a replantear el comportamiento de bajo nivel del modelo.

### El "Infierno de los Kernels" y Problemas de Memoria (OOM)

Durante el proceso de abstracción, surgió una considerable fricción técnica al intentar reescribir la operación matemática del *Selective Scan* mediante operaciones nativas de TensorFlow de alto nivel (como `tf.scan` o bucles `tf.while_loop`). Este enfoque ingenuo resultó en catastróficas caídas de rendimiento y severos errores de desbordamiento de memoria gráfica (Out Of Memory, OOM). Al procesar imágenes de alta resolución como las requeridas para lesiones dermatológicas, la VRAM explota rápidamente al tratar de almacenar el árbol computacional completo del *Selective Scan* para el *Autodiff* (flujo de propagación de gradientes o *Backward Pass*).

Para mitigar esta limitación técnica profunda, se recurrió a la programación explícita a nivel de hardware, encapsulando la lógica del barrido secuencial en un *Custom Op* hibridado en C++ y CUDA (denominado `MambaSelectiveScanLayer`), saltándose el grafo dinámico estándar de Python. Sin embargo, la interconexión entre las librerías compartidas iterativas (`.so`) y las precompiladas de Google Colab provocó fallos de enlazado y cruces de símbolos C++, un reto resuelto inyectando dinámicamente referencias al *runtime* global de TensorFlow.

> **NOTA PARA EL TFG:** Se deberá incluir aquí evidencia del *stack trace* o error de "undefined symbol" para ilustrar la madurez de la resolución del problema de enlazado e interoperabilidad de librerías en colab.

Finalmente, para consolidar operaciones puramente vectorizadas aledañas que evitaran la fragmentación de la memoria en la GPU, se aprovechó la capacidad del compilador XLA (`jit_compile=True`). Todo este proceso, aunque complejo y arduo metodológicamente, sentó unas bases sólidas de alta eficiencia sin penalizar la semántica matemática del modelo.

---
**Pendiente de revisión:**
- *¿Considerar añadir métricas de tiempo de compilación o VRAM ahorrada antes y después de XLA para robustecer el argumento técnico?*
- *Detallar visualmente o mediante extractos cortos el error real de Colab si se guardó.*
