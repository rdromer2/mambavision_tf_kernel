## 5.3 El Bloque MambaVision Mixer (Fase 6 actual)

El núcleo funcional diferenciador de la arquitectura híbrida adaptada recae en su bloque procesador denominado **MambaVision Mixer**. Este módulo se diseña bajo un paradigma paralelo en el cual confluyen los campos receptivos de textura local puramente convolucional con un receptor global contextual operado mediante SSMs.

> **[AÑADIR ESQUEMA AQUÍ]**  
> *(Nota: Insertar diagrama Draw.io o Lucidchart que muestre: 1) Tensor de Entrada -> 2) Proyecciones Lineales divisorias -> 3) Rama A: Mamba Selective Scan / Rama B: Convolución Simétrica -> 4) Concatenación cruzada.)*

### Proyecciones Lineales de Entrada

Al ingresar un tensor de características genérico, este es mapeado espacialmente a través de densas expansiones lineales y proyectado transversalmente. Se dividen las dimensiones de estado en múltiples ramales para permitir, de manera simultánea, operaciones independientes sobre sub-conjuntos discretos del tensor. Esta segmentación es fundamental para la selección dinámica contextual sin penalizar los requisitos computacionales de la representación visual.

### Convolución Espacial Simétrica ($X_{ssm}$ y $X_{sym}$)

La innovación morfológica radica en la bifurcación simétrica del tensor latente hacia dos ramales gemelos:
1. **Rama de Selección Global ($X_{ssm}$):** Donde el tensor es aplanado temporalmente en una única secuencia 1D (vector espacial) para ser procesado por la máquina de estados.
2. **Rama Convolucional Simétrica ($X_{sym}$):** Una matriz convolutiva preserva la geometría espacial explícita (`[H, W]`) y capta interacciones texturales colindantes, como los contornos granulares difusos presentes de forma natural en melanomas y nevos.

### Encapsulación del MambaSelectiveScanLayer (Núcleo C++/CUDA)

La fase culminante de la rama $X_{ssm}$ es el estado de escaneo o barrido continuo. En la implementación lograda en Keras, se encapsuló lógicamente bajo la clase de capa modular `MambaSelectiveScanLayer`. Esta clase actúa en Pyhton como un simple puente inyectable, pero invoca implícitamente la ejecución compilada paralelizada mediante XLA y las llamadas subyacentes explícitas codificadas de forma nativa en `.cu`. El tensor transita directamente en la memoria caché acelerada (VRAM compartida `__shared__` de CUDA) sin transferencias hacia el CPU para mantener latencias operativas del orden de microsegundos, resultando en reducciones drásticas del tiempo por métrica de *epoch*.

---
**Pendiente de revisión / Trabajo en curso:**
- *A nivel de repositorio, actualmente tenemos completada y funcional la capa `MambaSelectiveScanLayer` nativa en C++ y Python (`mamba_layer.py`). Sin embargo, queda pendiente integrar la estructura general completa del bloque "Mixer" que conecta la convolución simétrica (Rama X_sym) y las proyecciones lineales en el pipeline de Keras principal.*
