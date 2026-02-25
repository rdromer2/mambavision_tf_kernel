# **Informe y Guía Exhaustiva: Google Colab desde VS Code**

La extensión oficial googlecolab/colab-vscode permite conectar tu entorno local de Visual Studio Code directamente a los servidores de Google Colab. La ejecución y el consumo de recursos (CPU, memoria, GPUs, TPUs) ocurren en los servidores de Google, mientras usas la interfaz de tu editor local.

## **1\. Conceptos Clave (Cómo funciona por debajo)**

Para entender los comportamientos de la extensión, es necesario aclarar la terminología que usa Google Colab frente a lo que ves en VS Code. En la versión web de Colab, todo se agrupa bajo el concepto de "Entorno de ejecución" (Runtime). En VS Code, el proceso se divide en sus componentes reales:

* **Jupyter Server (El Servidor):** Es la aplicación backend principal que orquesta la experiencia. Actúa como intermediario entre tu VS Code (el frontend) y el motor de computación.  
* **Kernel:** Es el motor de computación real que ejecuta el código (ej. Python, Julia, R). Cuando ejecutas una celda, el código va al kernel, y este devuelve el resultado al servidor.  
* **Session (Sesión):** Es el enlace que conecta un notebook específico con una instancia de kernel específica.

*Analogía útil:* El Servidor es el gerente del restaurante que toma los pedidos del VS Code (el camarero), los delega a los Kernels (los cocineros) que leen los pedidos de una Sesión (el ticket de comanda) y asegura que el resultado vuelva a ti. En la web esto es automático; en VS Code tienes el control de conectar exactamente qué notebook a qué servidor/kernel.

## **2\. Instalación y Flujo de Trabajo Básico (La opción sencilla)**

Esta es la forma directa de ejecutar código en la nube en un único archivo.

1. Instala la extensión **Google Colab** en VS Code (requiere tener instalada la extensión de *Jupyter*).  
2. Abre o crea un archivo .ipynb.  
3. En la esquina superior derecha, haz clic en **Select Kernel \> Colab**.  
4. Verás dos opciones:  
   * **Auto Connect:** Se conecta a un servidor por defecto.  
   * **New Colab Server:** Te permite aprovisionar un tipo de máquina específico (ej. solicitar una GPU).  
5. Se te pedirá iniciar sesión con Google. El navegador se abrirá (debes permitir que VS Code abra enlaces externos). Tras la autorización OAuth, serás redirigido de vuelta a VS Code.  
6. Ejecuta una celda.

### **Comandos Básicos (Accesibles vía Ctrl+Shift+P / Cmd+Shift+P)**

* Colab: Mount Google Drive to Server...: Inserta el código drive.mount('/content/drive') en una celda.  
* Colab: Remove Server: Desconecta y elimina la máquina actual.  
* Colab: Sign Out: Cierra tu sesión de Google.

## **3\. Flujo de Trabajo Robusto (La mejor opción para largo plazo)**

Tener todo el código en un único archivo .ipynb dificulta el control de versiones y la reutilización de funciones, especialmente en proyectos de bioingeniería con preprocesamientos complejos. La opción más robusta requiere separar la lógica del proyecto.

### **Arquitectura recomendada**

Estructura tu proyecto en local así:

mi\_proyecto/  
├── data/                  \# Muestras pequeñas locales (ignorado en git)  
├── src/                   \# Lógica central en Python puro (.py)  
│   ├── \_\_init\_\_.py  
│   └── preprocesamiento.py   
├── notebooks/             \# Archivos Jupyter (.ipynb)  
│   └── 01\_analisis.ipynb  \# Conectado a Colab  
└── requirements.txt       

### **El problema del entorno remoto**

Al conectarte a Colab, el servidor remoto **no tiene tus archivos locales de la carpeta src/**. El servidor arranca con el sistema de archivos de Google (/content/).

Para resolver esto sin copiar/pegar código:

**Paso 1: Control de versiones**

Sube tu código local a un repositorio de GitHub/GitLab.

**Paso 2: Sincronización en el servidor de Colab**

En la primera celda de tu notebook (ejecutada en Colab), clona el repositorio y añade la ruta al sistema para importar tus módulos:

import sys  
import os

\# Clonar o actualizar el repositorio en la máquina de Google  
if not os.path.exists('/content/mi\_proyecto'):  
    \!git clone \[https://github.com/tu\_usuario/mi\_proyecto.git\](https://github.com/tu\_usuario/mi\_proyecto.git) /content/mi\_proyecto  
else:  
    \!cd /content/mi\_proyecto && git pull

\# Añadir al path para poder hacer 'import src.preprocesamiento'  
sys.path.append('/content/mi\_proyecto')

**Paso 3: Recarga automática**

En la segunda celda, activa la recarga para no tener que reiniciar el kernel cada vez que hagas un git pull con cambios nuevos:

%load\_ext autoreload  
%autoreload 2

from src.preprocesamiento import limpiar\_datos

**Paso 4: Datos Pesados**

Usa Colab: Mount Google Drive to Server... para acceder a tus datasets pesados (imágenes, CSVs masivos) directamente desde tu Drive sin subirlos en cada sesión.

## **4\. Funciones Experimentales (Avanzado)**

La guía de usuario detalla varias funciones experimentales que acercan la experiencia de Colab a un entorno de desarrollo local. Para activarlas, ve a los Ajustes de VS Code (Ctrl+,), busca "Colab" y marca las casillas correspondientes. **Nota:** Es necesario recargar la ventana de VS Code tras activarlas.

### **4.1. Server Mounting (Montaje del Servidor)**

Permite montar el sistema de archivos remoto de Colab (la carpeta /content/) directamente en tu espacio de trabajo (Workspace) de VS Code.

* **Uso:** Comando Colab: Mount Server To Workspace....  
* **Detalle:** Puedes ver, crear, editar y borrar archivos remotos desde el explorador de archivos de VS Code. Si un archivo se modifica fuera de VS Code (ej. por un script), debes pulsar el botón de refrescar en el explorador.  
* **Aplicación en el Flujo Robusto:** Útil para inspeccionar qué se ha descargado exactamente tras tu comando \!git clone o verificar las rutas de los archivos generados.

### **4.2. Colab Terminal**

Permite abrir una terminal bash directamente conectada a la máquina virtual de Colab.

* **Uso:** Comando Colab: Open Terminal.  
* **Aplicación en el Flujo Robusto:** En lugar de usar celdas con \! (ej. \!pip install \-r requirements.txt o \!git pull), puedes usar esta terminal para interactuar con el contenedor de Google exactamente igual que si estuvieras en una terminal local de Linux.

### **4.3. Uploading (Subida de archivos manual)**

Permite hacer clic derecho en archivos o carpetas de tu explorador local de VS Code y seleccionar Upload to Colab.

* **Detalle:** Si tienes varios servidores activos, te preguntará a cuál subirlo. El archivo va directamente a la carpeta /content/ del servidor remoto. Puede ser un sustituto rápido al git clone si solo necesitas enviar un script .py temporal, aunque para mantenimiento a largo plazo el flujo con Git sigue siendo superior.

### **4.4. Activity Bar (Barra de Actividad)**

Añade un icono de Colab en la barra lateral izquierda de VS Code. Desde ahí puedes visualizar tus servidores activos y explorar sus directorios /content sin necesidad de montar todo el servidor en tu espacio de trabajo principal.