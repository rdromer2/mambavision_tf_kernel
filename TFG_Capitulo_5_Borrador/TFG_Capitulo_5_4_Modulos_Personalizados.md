## 5.4 Módulos Personalizados (Custom Layers)

Para alcanzar el rendimiento máximo prescriptivo (SOTA) en la adaptación a flujos de trabajo sobre tejido biológico epitelial complejo, la estrategia arquitectónica demandó la inclusión y readaptación en TensorFlow de entidades y módulos que trascienden las funciones genéricas base de los bloques secuenciales.

### Función de Activación StarPRelu

El marco conceptual original requería una función de hiper-activación orientada no lineal que mitigase la muerte de gradientes en contextos visuales sin saturar el procesamiento escalar. En contraposición a las aproximaciones elementales tipo ReLU (Rectified Linear Unit), se incluyó heurística teórica para habilitar la adaptación matemática de **StarPRelu**.

La formulación de esta función introduce parámetros sutiles penalizadores controlables. Mientras las topologías genéricas cortan la no-linealidad abruptamente, StarPRelu escala la porción negativa gradualmente en base a hiper-parámetros inferidos directamente por la red durante el *backward pass*, otorgando mayor flexibilidad predictiva en gradientes semánticamente opacos, algo extremadamente beneficioso para detectar redes vasculares finas o glóbulos atípicos asimétricos.

### Módulo GFT (Gradient Feature Tokenization)

La limitación paradigmática de las secuencias SSM recae en su naturaleza analítica global; el barrido no prioriza inherentemente la resolución local a nivel micrométrico de manera direccional. Para resolver esta debilidad inherente de las extracciones Mamba aplicadas a visión dermatológica, se plantea la incorporación conceptual del módulo **GFT** (*Gradient Feature Tokenization*).

La adaptación del GFT en en el entorno enmarca el uso explícito de filtros convolucionales Sobel dentro de la etapa de "Tokenización" prematura. En lugar de derivar la imagen en píxeles crudos carentes de entropía interpretativa, se inyectan operadores direccionales laplacianos (bordes $x$ e $y$). Esta fusión de los gradientes en forma de *tokens* locales obliga semánticamente a la red Mamba a prestar atención persistente extrema a las sutiles interfaces micro-texturales ("retículos") sin difuminarlas al extender la receptividad global computacional en las fases finales.

---
**Pendiente de revisión / Trabajo en curso:**
- *Nota arquitectónica interna: Estas funciones personalizadas (StarPRelu y GFT) con operadores Sobel no han sido trasladadas a archivos físicos (scripts Keras) a fecha actua en este repositorio. Constituyen el siguiente paso imperativo en la hoja de ruta de la codificación una vez validado numéricamente el comportamiento del bloque Mixer central.*
- *Considerar incorporar o redactar las ecuaciones analíticas de StarPRelu empleando formato LaTeX dentro del documento final al exponer la matemática ante el tribunal para consolidar la justificación métrica.*
