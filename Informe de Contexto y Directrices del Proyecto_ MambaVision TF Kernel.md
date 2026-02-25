# **Informe de Contexto y Directrices del Proyecto: MambaVision TF Kernel**

## **1\. Objetivo Principal**

Reimplementar el mecanismo de *selective scan* del modelo MambaVision (originalmente escrito para PyTorch) como una operación personalizada (`OpKernel`) nativa en TensorFlow. Esto requiere aislar, adaptar, compilar y enlazar un kernel a muy bajo nivel utilizando C++ y CUDA, empaquetándolo finalmente como una librería independiente instalable.

## **2\. Arquitectura de Desarrollo y Flujo de Trabajo**

El proyecto utiliza una infraestructura distribuida para separar la escritura del código de la ejecución en el hardware especializado:

* **Entorno Local (Desarrollo):** Sistema Linux sin GPU dedicada. Se utiliza VS Code para la escritura del código (C++, CMake, CUDA), garantizando el análisis estático y el autocompletado nativo. El proyecto está encapsulado en un directorio propio, totalmente aislado de la base de código del proyecto principal para evitar contaminación de archivos temporales de compilación.  
* **Entorno de Ejecución (Nube):** Instancias efímeras de Google Colab con GPU (ej. T4 o superior).  
* **Puente de Sincronización:** El código fluye desde el entorno local a la nube estrictamente a través de un repositorio de control de versiones (Git). El entorno remoto ejecuta comandos de `pull` para obtener el código fresco y compila utilizando CMake.

## **3\. Directrices Estrictas para el Agente IA**

El agente IA que asista en este desarrollo debe adherirse imperativamente a las siguientes reglas operativas:

* **Nivel de Explicación y Detalle Técnico:** La dirección del proyecto se realiza desde una perspectiva de ingeniería biomédica. Quien lidera sabe programar y abstraer lógicas complejas, pero no se dedica exclusivamente al desarrollo de software. Por ello, el agente IA **siempre debe recordar y explicar los pequeños detalles técnicos**. No se debe dar por obvio ningún comportamiento de bajo nivel (ej. gestión de memoria, cálculo de *strides* en tensores, alineación de punteros, o el impacto de las banderas del compilador).  
* **Estrategia Dual de Resolución (Simple vs. Robusta):** Ante cualquier propuesta de arquitectura o código, el agente siempre debe proporcionar dos enfoques:  
  1. La opción fácil y sencilla para pruebas rápidas si es conveniente.  
  2. La **mejor opción, la más robusta y mantenible a largo plazo**, aunque sea más difícil de aplicar. El desarrollo siempre debe priorizar orientarse hacia esta opción óptima.  
* **Tono y Comunicación:** La IA debe abstenerse completamente de emitir juicios de valor. **No se deben incluir felicitaciones, ni validaciones positivas** sobre las decisiones o el progreso del usuario. Las respuestas deben ser estrictamente objetivas, técnicas y centradas en el problema a resolver.

## **4\. Mapa de Fases Técnicas**

1. **Fase de Validación (ABI y Entorno):** Configurar el archivo `CMakeLists.txt` base. Compilar una operación pasiva en C++ que se enlace con TensorFlow en Python, gestionando correctamente la bandera de compatibilidad `_GLIBCXX_USE_CXX11_ABI`.  
2. **Fase de Puente (Wrapper C++):** Desarrollar la interfaz CPU para leer los tensores N-dimensionales de TensorFlow, extrayendo los punteros directos a memoria, las dimensiones y validando la contigüidad antes de enviarlos a la gráfica.  
3. **Fase de Ejecución GPU (Forward Pass):** Adaptar el kernel original `.cu`. Gestionar la jerarquía de memoria de la GPU, asegurando la transferencia de los datos a la memoria SRAM compartida, configurando la cuadrícula de hilos (blocks/threads) y verificando los bloqueos de sincronización (`__syncthreads()`).  
4. **Fase de Aprendizaje (Backward Pass):** Escribir y registrar matemáticamente el kernel de los gradientes usando `@tf.RegisterGradient` para posibilitar el entrenamiento del modelo, no solo la inferencia.  
5. **Fase de Empaquetado:** Estructurar el código final para que el proyecto principal pueda consumirlo limpiamente como una dependencia externa.

