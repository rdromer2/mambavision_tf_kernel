import json

with open('compilador.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if line.strip() == "os.system('apt-get update -qq')":
                source[i] = "!apt-get update -qq\n"
            elif line.strip() == "os.system('apt-get install -y cmake')":
                source[i] = "!apt-get install -y cmake\n"
            elif line.strip() == "os.system('mkdir -p build')":
                source[i] = "!mkdir -p build\n"
            elif line.strip() == "os.system('cmake ..')":
                source[i] = "!cmake ..\n"
            elif line.strip() == "os.system('make')":
                source[i] = "!make\n"

        cell['source'] = source

with open('compilador.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
