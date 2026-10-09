"""Modelo de competencia entre dos plataformas.

El sistema es:

    dx/dt = r1 * x * (1 - x - a12 * y)
    dy/dt = r2 * y * (1 - y - a21 * x)
"""

import numpy as np


def campo(estado, a12, a21, r1, r2):
    """Evalúa el campo vectorial del sistema.

    Args:
        estado: Arreglo (x, y). También acepta mallas: x e y pueden ser arreglos.
        a12: Efecto de B sobre A.
        a21: Efecto de A sobre B.
        r1: Tasa de crecimiento de A.
        r2: Tasa de crecimiento de B.

    Returns:
        Arreglo (dx/dt, dy/dt).
    """
    x, y = estado
    return np.array([r1 * x * (1 - x - a12 * y), r2 * y * (1 - y - a21 * x)])


def jacobiana(estado, a12, a21, r1, r2):
    """Calcula la matriz jacobiana del sistema en un estado.

    Args:
        estado: Arreglo (x, y).
        a12, a21, r1, r2: Parámetros del modelo (ver `campo`).

    Returns:
        Matriz 2x2 de NumPy.
    """
    x, y = estado
    return np.array([
        [r1 * (1 - 2 * x - a12 * y), -r1 * a12 * x],
        [-r2 * a21 * y, r2 * (1 - 2 * y - a21 * x)],
    ])


def equilibrios(a12, a21):
    """Retorna los cuatro equilibrios del sistema.

    Args:
        a12, a21: Coeficientes de competencia (a12 * a21 != 1).

    Returns:
        Diccionario {nombre: arreglo (x*, y*)}. El interior puede tener
        coordenadas negativas (no factible).
    """
    den = 1 - a12 * a21
    return {
        "(0,0)": np.array([0.0, 0.0]),
        "(1,0)": np.array([1.0, 0.0]),
        "(0,1)": np.array([0.0, 1.0]),
        "interior": np.array([(1 - a12) / den, (1 - a21) / den]),
    }


def clasificar(J):
    """Clasifica un equilibrio a partir de su jacobiana.

    Args:
        J: Matriz jacobiana 2x2 evaluada en el equilibrio.

    Returns:
        Diccionario con traza, determinante, discriminante, valores
        propios y tipo de equilibrio.
    """
    tau, delta = np.trace(J), np.linalg.det(J)
    disc = tau**2 - 4 * delta
    if delta < 0:
        tipo = "silla"
    elif tau == 0:
        tipo = "centro"
    else:
        forma = "nodo" if disc >= 0 else "foco"
        tipo = f"{forma} {'estable' if tau < 0 else 'inestable'}"
    return {
        "τ": tau,
        "Δ": delta,
        "τ²-4Δ": disc,
        "λ": np.linalg.eigvals(J),
        "tipo": tipo,
    }


def euler(f, x0, h, T):
    """Integra dx/dt = f(x) con el método de Euler explícito.

    Args:
        f: Campo vectorial, función f(x) que retorna un arreglo.
        x0: Estado inicial.
        h: Paso de integración.
        T: Tiempo final.

    Returns:
        Tupla (t, X): tiempos de tamaño n+1 y estados de forma (n+1, dim).
    """
    n = int(round(T / h))
    X = np.empty((n + 1, len(x0)))
    X[0] = x0
    for k in range(n):
        X[k + 1] = X[k] + h * f(X[k])
    return np.linspace(0, n * h, n + 1), X


def rk4(f, x0, h, T):
    """Integra dx/dt = f(x) con el método de Runge-Kutta de cuarto orden.

    Args:
        f: Campo vectorial, función f(x) que retorna un arreglo.
        x0: Estado inicial.
        h: Paso de integración.
        T: Tiempo final.

    Returns:
        Tupla (t, X): tiempos de tamaño n+1 y estados de forma (n+1, dim).
    """
    n = int(round(T / h))
    X = np.empty((n + 1, len(x0)))
    X[0] = x0
    for k in range(n):
        x = X[k]
        k1 = f(x)
        k2 = f(x + h / 2 * k1)
        k3 = f(x + h / 2 * k2)
        k4 = f(x + h * k3)
        X[k + 1] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.linspace(0, n * h, n + 1), X


def destino(f, X0, h, t_max, atractores, tol=1e-4):
    """Determina a qué atractor converge cada estado inicial usando RK4.

    Integra todos los estados a la vez y marca cada uno cuando su distancia
    a un atractor es menor que `tol`.

    Args:
        f: Campo vectorial f(x) que acepta arreglos de forma (2, N).
        X0: Estados iniciales de forma (N, 2).
        h: Paso de integración.
        t_max: Tiempo máximo de simulación.
        atractores: Lista de equilibrios estables, cada uno de forma (2,).
        tol: Distancia que define la convergencia.

    Returns:
        Tupla (etiqueta, t_conv): índice del atractor alcanzado (-1 si no
        converge antes de `t_max`) y tiempo de convergencia de cada estado.
    """
    X = np.array(X0, dtype=float).T
    n = X.shape[1]
    etiqueta = np.full(n, -1)
    t_conv = np.full(n, np.nan)
    t = 0.0
    activos = np.arange(n)
    while t < t_max and activos.size:
        x = X[:, activos]
        k1 = f(x)
        k2 = f(x + h / 2 * k1)
        k3 = f(x + h / 2 * k2)
        k4 = f(x + h * k3)
        X[:, activos] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        for i, a in enumerate(atractores):
            dist = np.linalg.norm(X[:, activos] - np.reshape(a, (2, 1)), axis=0)
            llegaron = activos[dist < tol]
            etiqueta[llegaron] = i
            t_conv[llegaron] = t
        activos = activos[etiqueta[activos] == -1]
    return etiqueta, t_conv
