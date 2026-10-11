"""Modelo de lago con reciclaje interno de nutrientes y señales de alerta temprana.

El modelo es:

    dx/dt = f(x; a) = a - b * x + x**2 / (x**2 + 1)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.stats import kendalltau


def f(x, a, b):
    """Evalúa el campo dx/dt del lago.

    Args:
        x: Concentración de nutrientes (escalar o arreglo).
        a: Carga externa de nutrientes.
        b: Tasa de pérdida.

    Returns:
        Valor de dx/dt.
    """
    return a - b * x + x**2 / (x**2 + 1)


def g(x, b):
    """Carga de equilibrio a = g(x) = b * x - x**2 / (x**2 + 1)."""
    return b * x - x**2 / (x**2 + 1)


def dg(x, b):
    """Derivada g'(x) = b - 2x / (x**2 + 1)**2."""
    return b - 2 * x / (x**2 + 1) ** 2


def raices(fun, x_min=0.0, x_max=5.0, n=50001):
    """Encuentra las raíces de una función en un intervalo.

    Busca cambios de signo en una malla y refina cada uno con `brentq`.

    Args:
        fun: Función escalar de una variable.
        x_min: Extremo inferior del intervalo.
        x_max: Extremo superior del intervalo.
        n: Número de puntos de la malla.

    Returns:
        Arreglo ordenado con las raíces encontradas.
    """
    xs = np.linspace(x_min, x_max, n)
    v = fun(xs)
    idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    r = [brentq(fun, xs[i], xs[i + 1]) for i in idx] + list(xs[v == 0])
    return np.sort(r)


def equilibrios(a, b, x_max=5.0):
    """Retorna los equilibrios x* que cumplen f(x*; a) = 0, ordenados."""
    return raices(lambda x: f(x, a, b), 0.0, x_max)


def rampa_determinista(a0, v, T, x0, b, dt=0.1):
    """Integra con RK4 el lago con carga variable a(t) = a0 + v * t.

    Args:
        a0: Carga inicial.
        v: Velocidad de cambio de la carga (puede ser negativa).
        T: Tiempo final.
        x0: Estado inicial.
        b: Tasa de pérdida.
        dt: Paso de integración.

    Returns:
        Tupla (t, a, x) de arreglos.
    """
    n = int(round(T / dt))
    t = np.linspace(0, n * dt, n + 1)
    x = np.empty(n + 1)
    x[0] = x0
    for k in range(n):
        tk, xk = t[k], x[k]
        a1, a2, a3 = a0 + v * tk, a0 + v * (tk + dt / 2), a0 + v * (tk + dt)
        k1 = f(xk, a1, b)
        k2 = f(xk + dt / 2 * k1, a2, b)
        k3 = f(xk + dt / 2 * k2, a2, b)
        k4 = f(xk + dt * k3, a3, b)
        x[k + 1] = xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, a0 + v * t, x


def euler_maruyama(x0, a_t, b, sigma, dt, n_real, rng):
    """Simula `n_real` realizaciones de dx = f(x; a(t)) dt + sigma dW.

    Args:
        x0: Estado inicial (igual para todas las realizaciones).
        a_t: Arreglo con la carga en cada paso; define el número de pasos.
        b: Tasa de pérdida.
        sigma: Intensidad del ruido.
        dt: Paso de tiempo.
        n_real: Número de realizaciones.
        rng: Generador `numpy.random.Generator`.

    Returns:
        Arreglo (n_real, len(a_t)) con las trayectorias.
    """
    X = np.empty((n_real, len(a_t)))
    X[:, 0] = x0
    sq = sigma * np.sqrt(dt)
    for k in range(len(a_t) - 1):
        X[:, k + 1] = X[:, k] + f(X[:, k], a_t[k], b) * dt + sq * rng.standard_normal(n_real)
    return X


def detrend(x, ancho):
    """Elimina la tendencia restando un suavizado gaussiano.

    Args:
        x: Serie de tiempo.
        ancho: Desviación estándar del kernel gaussiano, en número de puntos.

    Returns:
        Residuos x - tendencia.
    """
    return x - gaussian_filter1d(x, ancho, mode="nearest")


def indicadores_moviles(r, ventana):
    """Calcula la varianza y la autocorrelación de retardo uno en ventanas móviles.

    Args:
        r: Serie de residuos.
        ventana: Tamaño de la ventana, en número de puntos.

    Returns:
        Tupla (var, ac1). El elemento i corresponde a la ventana que
        termina en el punto i + ventana - 1.
    """
    c = lambda z: np.concatenate([[0.0], np.cumsum(z)])
    w = ventana
    s1, s2 = c(r), c(r**2)
    suma = s1[w:] - s1[:-w]
    var = (s2[w:] - s2[:-w]) / w - (suma / w) ** 2

    # Pares (r_t, r_{t+1}) dentro de cada ventana: w - 1 pares
    a, bb = r[:-1], r[1:]
    sa, sb, sab = c(a), c(bb), c(a * bb)
    sa2, sb2 = c(a**2), c(bb**2)
    m = w - 1
    Sa, Sb = sa[m:] - sa[:-m], sb[m:] - sb[:-m]
    cov = (sab[m:] - sab[:-m]) / m - Sa * Sb / m**2
    va = (sa2[m:] - sa2[:-m]) / m - (Sa / m) ** 2
    vb = (sb2[m:] - sb2[:-m]) / m - (Sb / m) ** 2
    ac1 = cov / np.sqrt(va * vb)
    return var, ac1


def rama_clara(a, b, x_pliegue):
    """Equilibrio de aguas claras x*(a) para a menor que la carga de colapso.

    Invierte a = g(x) en el tramo creciente [0, x_pliegue] por interpolación.

    Args:
        a: Carga (escalar o arreglo).
        b: Tasa de pérdida.
        x_pliegue: Posición del pliegue de colapso (primer cero de g').

    Returns:
        x*(a) sobre la rama de aguas claras.
    """
    xs = np.linspace(0, x_pliegue, 20001)
    return np.interp(a, g(xs, b), xs)


def tendencia_kendall(serie, ancho, submuestreo=20):
    """Calcula el tau de Kendall de la varianza y la autocorrelación móviles.

    Elimina la tendencia con `detrend`, calcula los indicadores con una
    ventana de la mitad de la serie y evalúa su tendencia contra el tiempo.

    Args:
        serie: Serie de tiempo previa a la transición.
        ancho: Ancho del suavizado gaussiano, en número de puntos.
        submuestreo: Se usa un punto de cada `submuestreo` del indicador.

    Returns:
        Arreglo (tau_var, p_var, tau_ac1, p_ac1).
    """
    var, ac1 = indicadores_moviles(detrend(serie, ancho), len(serie) // 2)
    i = np.arange(len(var))[::submuestreo]
    kv, ka = kendalltau(i, var[::submuestreo]), kendalltau(i, ac1[::submuestreo])
    return np.array([kv.statistic, kv.pvalue, ka.statistic, ka.pvalue])
