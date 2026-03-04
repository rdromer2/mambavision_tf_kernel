# 📘 MambaVision TF Kernel: Project Context

Este documento sirve como referencia rápida superior para el agente IA al iniciar cualquier sesión de trabajo.

## 1. Objetivo Principal
Reimplementar el mecanismo de **selective scan** del modelo MambaVision (originalmente en PyTorch) como una operación personalizada (`OpKernel`) nativa en TensorFlow.
Esto forma un puente de muy bajo nivel (C++ y CUDA) empaquetado como una librería dinámica instalable e independiente, altamente optimizada para la ejecución en hardware gráfico.

## 2. Arquitectura de Desarrollo
El proyecto opera bajo un modelo distribuido local-nube:
- **Entorno Local (Desarrollo):** Sistema Linux con VS Code, enfocado en el análisis estático y escritura pura (C++, CMake, Python). Estrictamente aislado del sistema principal para asegurar portabilidad.
- **Entorno de Ejecución (Nube):** Entornos efímeros en Google Colab con hardware acelerado (GPU T4, A100).
- **Sincronización:** Flujo unidireccional puramente a través de Git (`https://github.com/rdromer2/mambavision_tf_kernel.git`). El entorno en la nube descarga los repositorios en cada sesión mediante `git clone` / `git pull`.

## 3. Directrices Estrictas de la IA
- **Nivel Técnico y de Explicación:** El desarrollador jefe tiene perfil biomédico; entiende ingeniería, pero **no presumas nada sobre aspectos informáticos de ultra-bajo nivel**. La IA debe detallar *siempre* cómo funciona tras bambalinas la VRAM, *strides* de memoria, alineación, y configuraciones (flags) del compilador.
- **Comunicación Estricta:** Las interacciones deben carecer por completo de lenguaje asertivo inútil. Sin juicios de valor, sin felicitaciones, directo al grano y enfocado puramente a nivel de ingeniería.

## 4. Fases Técnicas del Proyecto
1. **Fase de Validación:** Resolver los choques de compatibilidad ABI vs Entorno (GCC vs pip TensorFlow); asegurar compilación base de Op C++ conectando con Python.
2. **Fase Puente (C++ Wrapper CPU):** Desarrollar la inferencia dimensional para capturar las matrices N-dimensionales, validar los *shapes* y obtener los punteros limpios de la VRAM.
3. **Fase Ejecución GPU (Forward Pass):** Traducción y adaptador del kernel `.cu` que transfiere memoria HBM -> SRAM, ejecutando los hilos NVIDIA paralelizados sincrónicamente.
4. **Fase Aprendizaje (Backward Pass):** Adaptación del paso de gradientes con `@tf.RegisterGradient` para propagación atrás durante el entrenamiento.
5. **Fase de Empaquetado:** Módulo final preparado para consumo limpio desde un script externo.
