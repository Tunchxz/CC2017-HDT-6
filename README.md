# Hoja de Trabajo 6

Esta actividad analiza la estabilidad de dos sistemas dinámicos (grupo 10):

- **Task 1 y Task 2** (`task1_2.ipynb`): modelo de competencia entre dos plataformas digitales. Clasifica los equilibrios, dibuja nulclinas y el retrato de fases, compara Euler y Runge-Kutta 4, calcula el mapa de cuencas y evalúa intervenciones. Las funciones están en `competencia.py`.

- **Task 3** (`task3.ipynb`): modelo de lago con punto de no retorno. Calcula el diagrama de equilibrios y la histéresis, simula con Euler-Maruyama y evalúa señales de alerta temprana (varianza y autocorrelación). Las funciones están en `lago.py`.

## Cómo Ejecutar

Requisitos: Python 3.14 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 1 y Task 2:

```bash
jupyter notebook task1_2.ipynb
```

Task 3:

```bash
jupyter notebook task3.ipynb
```
