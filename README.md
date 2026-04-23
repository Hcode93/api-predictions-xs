# API PREDICCTIONS xs

### Requisitos Previos
Instalar Python
Si aún no tienes Python instalado, sigue estos pasos según tu sistema operativo:

- Windows: Descarga el instalador desde python.org. Al instalar, asegúrate de marcar la casilla "Add Python to `PATH".
- macOS: Puedes usar `brew install python` o descargar el instalador oficial.
- Linux (Ubuntu/Debian): Ejecuta `sudo apt update && sudo apt install python3 python3-venv`.

Verificación: Abre una terminal y escribe python --version (o python3 --version). Deberías ver una versión 3.8 o superior.

### Repositorio
#### Opción 1

1. **Hacer un Fork:** Haz clic en el botón "Fork" en la esquina superior derecha de este repositorio para crear una copia en tu cuenta personal.
2. Clonar **TU PROPIO Fork**

#### Opción 2

Descargar el contenido del repositorio (sin seguimiento en git)

### Configuración del entorno

#### Crear el entorno virtual:

```
Bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

#### Activar el entorno

```
Bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### Instalar dependencias

```
Bash
pip install -r requirements.txt
```

### Levantar el API

Una vez que tengas el archivo `.joblib` en la carpeta raíz, ejecuta el servidor:
```
uvicorn main:app --reload
```

Con el servidor corriendo, entrar a: http://127.0.0.1:8000/docs