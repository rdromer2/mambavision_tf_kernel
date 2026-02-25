import json
import os

with open('compilador.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if line.strip() == "os.system(f\"git clone {GITHUB_REPO} {PROJECT_PATH}\")" or line.strip() == '!git clone {GITHUB_REPO} {PROJECT_PATH}':
                source[i] = "    os.system(f'git clone {GITHUB_REPO} {PROJECT_PATH}')\n"
            elif line.strip() == "os.system(f\"cd {PROJECT_PATH} \\!cd {PROJECT_PATH} && git pull\\!cd {PROJECT_PATH} && git pull git pull\")" or line.strip() == '!cd {PROJECT_PATH} && git pull':
                source[i] = "    os.system(f'cd {PROJECT_PATH} && git pull')\n"
            elif "os.system(f\"cd {PROJECT_PATH}" in line or "!cd {PROJECT_PATH} && git pull" in line:
                 source[i] = "    os.system(f'cd {PROJECT_PATH} && git pull')\n"
            elif "os.system(f\"git clone {GITHUB_REPO}" in line or "!git clone {GITHUB_REPO} {PROJECT_PATH}" in line:
                 source[i] = "    os.system(f'git clone {GITHUB_REPO} {PROJECT_PATH}')\n"

        cell['source'] = source

with open('compilador.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
