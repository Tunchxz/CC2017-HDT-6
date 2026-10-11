# Bitácora de IA

Herramienta utilizada: Claude Code (modelo Opus 5.5).

## Task 1 y Task 2

### Prompt

```text
Implementa el análisis del modelo de competencia de Lotka-Volterra en un módulo `competencia.py` y
un Jupyter Notebook `task1_2.ipynb`. Usa NumPy para el modelo y los integradores, matplotlib solo
para graficar y SciPy únicamente para `solve_ivp` como referencia de comparación. Euler y RK4 deben
ser implementaciones propias.

    dx/dt = r1·x·(1 - x - a12·y)
    dy/dt = r2·y·(1 - y - a21·x)

Parámetros: a12 = 1.4, a21 = 1.5, r1 = 1, r2 = 0.9. Estados de referencia: P1 = (0.05, 0.05),
P2 = (0.80, 0.30), P3 = (0.30, 0.80) y P4 = (0.60, 0.60). Fija la semilla 42.

Estructura del módulo:

1. `campo(estado, a12, a21, r1, r2)` que acepte un estado (x, y) o mallas de NumPy y devuelva
   (dx/dt, dy/dt), para reutilizarlo en el campo vectorial y en simulaciones vectorizadas.
2. `jacobiana(estado, ...)`, `equilibrios(a12, a21)` con los cuatro equilibrios, y
   `clasificar(J)` que devuelva traza, determinante, discriminante, valores propios y tipo.
3. `euler(f, x0, h, T)` y `rk4(f, x0, h, T)` que reciban el campo f(x), el estado inicial, el paso
   y el tiempo final, y devuelvan (t, X) con la trayectoria completa de forma (n+1, 2).
4. `destino(f, X0, h, t_max, atractores, tol)` que integre con RK4 muchos estados a la vez
   (arreglo de forma (2, N)) y marque cada uno cuando su distancia a un atractor sea menor que
   `tol`, sin guardar las trayectorias.

Celdas del notebook:

1. Tabla de equilibrios con τ, Δ, τ² - 4Δ, λ y tipo, y su ubicación en el plano traza-determinante
   con la parábola τ² = 4Δ. Guarda la figura en `figuras/`.
2. Nulclinas x + 1.4y = 1 y y + 1.5x = 1 con los ejes, los equilibrios marcados y una flecha por
   región según los signos de (dx/dt, dy/dt). Guarda la figura en `figuras/`.
3. Error en t = 10 desde P2 de Euler y RK4 contra `solve_ivp` (DOP853, rtol = atol = 1e-12) para
   h ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}. Gráfica log-log, pendiente por `np.polyfit` y pendientes
   locales entre pasos consecutivos.
4. Retrato de fases en [0, 1.5]²: `quiver` con flechas normalizadas, nulclinas, al menos doce
   trayectorias RK4 (h = 0.01, T = 40) que incluyan estados a 0.01 de los ejes, equilibrios
   coloreados por tipo y P1 a P4 resaltados. Imprime el estado final de cada P.
5. Mapa de cuencas con una malla 60 × 60 en [0.01, 1.5]² usando `destino` (h = 0.05,
   t_max = 500, tol = 1e-4). Reporta la fracción de cada cuenca. Estima la frontera por bisección
   sobre diez segmentos horizontales y = c (tolerancia 1e-6) y compara su pendiente cerca de la
   silla con la del vector propio de valor propio negativo.
6. Perturbaciones de 1e-4 en ocho direcciones dentro del cuadrante para (1,0) y (0,1). Ajusta una
   recta a ln‖x(t) - x*‖ en la ventana [1e-6, 1e-4] y en una ventana tardía [1e-11, 1e-8], y
   tabula ambas tasas con su error relativo frente a min|Re(λ)|.
7. Paso máximo de Euler por bisección desde un estado a 1e-3 del equilibrio: converge si tras
   20000 pasos la distancia es menor que 1e-8. Suprime los overflow con `np.errstate`. Compáralo
   con 2/|λ|.
8. Δ mínimo de usuarios añadidos a y en P2 que cambia el destino, por bisección sobre `destino`,
   graficado sobre el mapa de cuencas. Simulaciones con a21 = 0.9 y 1.1 desde (0.99, 0.01) y P2,
   y tiempos de llegada a (0,1) para Δ ∈ {1.01, 1.1, 1.5}·Δmin y a21 ∈ {0.99, 0.95, 0.9, 0.8}.
```

### ¿Por qué funcionó?

Funcionó porque el prompt fija los parámetros del grupo, los estados de referencia y las tolerancias numéricas de cada experimento. No queda margen para elegir pasos, tiempos finales o criterios de convergencia distintos entre celdas, y basta revisar los imports para confirmar que Euler y RK4 no dependen de SciPy, que solo aparece como referencia.

También define la firma de los integradores antes de escribir código. Al recibir un campo `f(x)` genérico y devolver la trayectoria completa, `euler` y `rk4` se reutilizan en la convergencia, el retrato de fases, las perturbaciones y las simulaciones con otro $\alpha_{21}$, cambiando solo la función que se pasa. La decisión clave es separar `destino` como una versión vectorizada de RK4 que no guarda trayectorias, con eso las 3600 simulaciones del mapa de cuencas y cada evaluación de las bisecciones corren en segundos y sin problemas de memoria.

Finalmente, los casos que suelen fallar se resuelven dentro del propio prompt. Pedir `campo` compatible con mallas permite dibujar el campo vectorial sin ciclos. El criterio de 20000 pasos para Euler evita confundir una convergencia lenta cerca del límite con una divergencia, y `np.errstate` evita que los pasos inestables llenen la salida de advertencias. Y pedir una ventana tardía además de la indicada permite separar el transitorio inicial del decaimiento a tiempos largos.

## Task 3

### Prompt

```text
Implementa el modelo de lago con reciclaje de nutrientes en un módulo `lago.py` y un Jupyter
Notebook `task3.ipynb`. Usa NumPy para el modelo y los integradores, SciPy solo para `brentq`,
`gaussian_filter1d` y `kendalltau`, y matplotlib solo para graficar.

    dx/dt = f(x; a) = a - b·x + x²/(x² + 1),   con b = 0.52
    a = g(x) = b·x - x²/(x² + 1)

Estructura del módulo:

1. `f(x, a, b)`, `g(x, b)` y `dg(x, b)` vectorizadas.
2. `raices(fun, x_min, x_max, n)` que busque cambios de signo en una malla, refine con `brentq` e
   incluya las raíces que caen exactamente en un punto de la malla. `equilibrios(a, b)` la usa
   sobre f.
3. `rampa_determinista(a0, v, T, x0, b, dt)` con RK4 propio para a(t) = a0 + v·t, evaluando a en
   t, t + dt/2 y t + dt dentro de cada paso.
4. `euler_maruyama(x0, a_t, b, sigma, dt, n_real, rng)` que simule todas las realizaciones a la
   vez, donde el arreglo `a_t` define la carga en cada paso.
5. `detrend(x, ancho)` que reste un suavizado gaussiano, e `indicadores_moviles(r, ventana)` que
   calcule la varianza y la autocorrelación de retardo uno en ventanas móviles con sumas
   acumuladas, sin ciclos sobre las ventanas.
6. `rama_clara(a, b, x_pliegue)` que invierta a = g(x) por interpolación en [0, x_pliegue], y
   `tendencia_kendall(serie, ancho, submuestreo)` que devuelva τ y p de ambos indicadores con
   una ventana de la mitad de la serie.

Celdas del notebook:

1. Ceros de g' (pliegues), a_col y a_rec y el ancho a_col - a_rec.
2. Diagrama de equilibrios (x* contra a) con la rama inestable en otro color y estilo.
3. Rampas deterministas de 0.005 a 0.09 y de regreso con v = 1e-5 y v = 1e-4. Reporta el a en
   que x cruza 0.6 en cada sentido y grafica ambas rampas sobre el diagrama.
4. Una realización con σ = 0.01, dt = 0.05, a0 = 0.5·a_col y v = (a_col - a0)/2000, desde el
   equilibrio de aguas claras. El salto es el primer índice en que x supera el pliegue de colapso.
5. Sobre la serie previa al salto: detrend con ancho de 1000 puntos y ventana de la mitad de la
   serie. Grafica varianza y AC1 con la teoría σ²/(2k) y e^(-k·dt), donde k = g'(x*(a(t))).
6. Una función `experimento(sigma, t_col, rampa, semilla)` que corra 100 realizaciones y devuelva
   la fracción con τ > 0 y p < 0.05 para cada indicador y para ambos, el τ mediano y el a del
   salto relativo a a_col. Con `rampa=False` la carga queda fija en a0 durante t_col.
7. Llama a `experimento` para la rampa base, el control con carga constante, σ = 0.02 y subida en
   1000, cada uno con su control y una semilla distinta derivada de 42.
```

### ¿Por qué funcionó?

Funcionó porque el prompt entrega el modelo, el valor de $b$ y todos los parámetros de las simulaciones estocásticas como dato duro. No queda margen para escoger otra velocidad de rampa, otro paso de tiempo o un estado inicial distinto, y la regla para detectar el salto (superar el pliegue de colapso) queda definida antes de ver cualquier serie.

También fija las funciones del módulo para que cada celda sea corta. Encontrar raíces por cambios de signo con `brentq` sirve tanto para los pliegues como para los equilibrios, y el caso de la raíz exacta en la malla se menciona porque en $a = 0$ el equilibrio $x = 0$ cae justo sobre ella. La decisión clave es concentrar las 100 realizaciones y todo el procesamiento en `experimento`, con eso la rampa, el control y los dos casos de sensibilidad son llamadas de una línea que solo cambian $\sigma$, el tiempo de subida o la carga fija.

Finalmente, el costo computacional se controla dentro del propio prompt. Euler-Maruyama vectorizado simula las 100 realizaciones con un solo ciclo en el tiempo, y las sumas acumuladas calculan los indicadores sobre ventanas de miles de puntos sin recorrer cada ventana. Y el submuestreo del indicador antes de `kendalltau` mantiene el notebook completo en unos dos minutos.
