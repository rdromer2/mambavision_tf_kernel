import json

with open('compilador.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if len(source) > 0 and source[0].startswith("# Instalar CMake"):
            # Reemplazar la celda de instalar cmake por la del compilador en python
            cell['source'] = [
                "import os\n",
                "import tensorflow as tf\n",
                "\n",
                "print(\"Compilando usando el g++ y ABI nativos de tf.sysconfig...\")\n",
                "TF_CFLAGS = ' '.join(tf.sysconfig.get_compile_flags())\n",
                "TF_LFLAGS = ' '.join(tf.sysconfig.get_link_flags())\n",
                "\n",
                "os.chdir(PROJECT_PATH)\n",
                "compile_cmd = f\"g++ -std=c++17 -shared zero_out.cc -o zero_out.so -fPIC {TF_CFLAGS} {TF_LFLAGS} -O2\"\n",
                "print(\"Ejecutando:\", compile_cmd)\n",
                "result = os.system(compile_cmd)\n",
                "if result == 0:\n",
                "    print(\"\\n✅ Compilación de zero_out.so finalizada con éxito.\")\n",
                "else:\n",
                "    print(f\"\\n❌ Error en la compilación. Código: {result}\")\n"
            ]
        elif len(source) > 0 and source[0].startswith("# Ejecutar la compilación"):
            # Vaciar la celda de Make antigua que ya no sirve
            cell['source'] = [
                "# Make reemplazado por compilador nativo de la celda anterior.\n",
                "pass\n"
            ]
        elif len(source) > 0 and source[0].startswith("import tensorflow as tf"):
            # Ajustar la celda 4 para que busque el .so en la raiz y no en build/
            for i, line in enumerate(source):
                if line.startswith("    zero_out_module = tf.load_op_library("):
                    source[i] = "    zero_out_module = tf.load_op_library(f'{PROJECT_PATH}/zero_out.so')\n"

with open('compilador.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
