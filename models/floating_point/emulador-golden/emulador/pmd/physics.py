"""Física del PMD: rotación de polarización y retardo diferencial (DGD), en SI.

Señal: forma (2, n), filas X e Y; el tiempo es el último eje.

Una sección de fibra birrefringente (docstring de ``blocks/pmd.py``):
    J(w) = R(theta, phi) @ diag( exp(+j*w*tau/2), exp(-j*w*tau/2) ) @ R(theta, phi)^H
  1. R^H : proyecta la señal sobre los ejes rápido/lento de la fibra.
  2. D   : un eje se adelanta tau/2 y el otro se atrasa tau/2.
  3. R   : vuelve a expresar la señal en X e Y.

Por qué R tiene dos ángulos (theta, phi):
  En la esfera de Poincaré, el eje rápido de la sección es
      s = (cos 2theta, sin 2theta cos phi, sin 2theta sin phi).
  Con solo theta (phi = 0) todos los ejes quedan en un plano: el DGD total es una
  caminata aleatoria en 2D y sale con distribución de Rayleigh, NO Maxwelliana.
  Con ejes uniformes en toda la esfera (``random_angles``) la caminata es 3D y el
  DGD total es Maxwelliano, como en una fibra real.
"""
from __future__ import annotations

import numpy as np


def rotation(theta: float, phi: float = 0.0) -> np.ndarray:
    """Matriz 2x2 unitaria que orienta los ejes de la fibra: R(theta, phi).

        R = [[ cos θ,            −sin θ · e^{−jφ} ],
             [ sin θ · e^{+jφ},   cos θ           ]]

    Con phi = 0 es la rotación real:
        y_X = cos θ · x_X  −  sin θ · x_Y
        y_Y = sin θ · x_X  +  cos θ · x_Y
    phi agrega un desfasaje entre X e Y (ejes elípticos en vez de lineales).

    - theta: ángulo de rotación, en radianes.
    - phi: fase entre X e Y, en radianes.
    - return: array (2, 2), complejo. Su inversa es R.conj().T.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s * np.exp(-1j * phi)],
                     [s * np.exp(1j * phi), c]])


def random_angles(rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Sortea (theta, phi) de k secciones, con el eje uniforme en la esfera de Poincaré.

    Para que el eje s sea uniforme en la esfera, cos 2theta tiene que ser uniforme
    en [-1, 1] (no theta uniforme) y phi uniforme en [0, 2pi).

    - rng: generador de numpy (``np.random.default_rng(seed)``).
    - k: cantidad de secciones.
    - return: (thetas, phis), dos arrays de largo k.
    """
    thetas = 0.5 * np.arccos(rng.uniform(-1, 1, k))
    phis = rng.uniform(0, 2 * np.pi, k)
    return thetas, phis


def dgd_delay(x: np.ndarray, fs: float, tau: float) -> np.ndarray:
    """Retardo diferencial entre las dos filas, aplicado en frecuencia.

    La fila 0 se adelanta tau/2 (exp(+j*w*tau/2)) y la fila 1 se atrasa tau/2
    (exp(-j*w*tau/2)). Todos los factores tienen módulo 1: no cambia la potencia.
    La FFT es circular: lo que sale por un extremo entra por el otro.

    - x: señal (2, n).
    - fs: frecuencia de muestreo, en Hz.
    - tau: DGD, en segundos.
    - return: señal (2, n).
    """
    n = x.shape[-1]
    w = 2 * np.pi * np.fft.fftfreq(n, 1 / fs)  # frecuencia angular de cada bin
    X = np.fft.fft(x, axis=-1)  # FFT por polarización
    H = np.stack([np.exp(+1j * w * tau / 2),   # fila 0: se adelanta tau/2
                  np.exp(-1j * w * tau / 2)])  # fila 1: se atrasa tau/2
    return np.fft.ifft(H * X, axis=-1)


def pmd_section(x: np.ndarray, fs: float, tau: float, theta: float, phi: float = 0.0) -> np.ndarray:
    """Una sección de PMD: y = R @ D(tau) @ R^H @ x, con R = R(theta, phi).

    R no depende de la frecuencia, así que se aplica en el tiempo. R es compleja,
    así que R^H = R.conj().T (con phi = 0 coincide con R.T).

    - x: señal (2, n), filas X e Y.
    - fs: frecuencia de muestreo, en Hz.
    - tau: DGD de la sección, en segundos.
    - theta, phi: orientación de los ejes de la fibra (ver ``rotation``), en radianes.
    - return: señal (2, n).
    """
    R = rotation(theta, phi)
    return R @ dgd_delay(R.conj().T @ x, fs, tau)


# --------------------------------------------------------------------------
# Matriz de Jones y DGD total (sin pasar señal)
# --------------------------------------------------------------------------
def jones_matrix(w: float, tau: float, thetas: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """Matriz de Jones total J_total(w) = J_K(w) @ ... @ J_1(w), en una frecuencia.

    - w: frecuencia angular, en rad/s (0 = portadora).
    - tau: DGD por sección, en segundos.
    - thetas, phis: ángulos de cada sección (largo K).
    - return: array (2, 2), complejo y unitario.
    """
    D = np.diag([np.exp(1j * w * tau / 2), np.exp(-1j * w * tau / 2)])
    J = np.eye(2, dtype=complex)
    for theta, phi in zip(thetas, phis):
        R = rotation(theta, phi)
        J = R @ D @ R.conj().T @ J  # la sección 1 se aplica primero
    return J


def total_dgd(tau: float, thetas: np.ndarray, phis: np.ndarray) -> float:
    """DGD total de la cascada en la portadora, por Jones Matrix Eigenanalysis (JME).

    Se compara J en dos frecuencias muy cercanas, w = 0 y w = dw:
        T = J(dw) @ J(0)^H,   autovalores rho_1, rho_2
        DGD = |arg(rho_1 / rho_2)| / dw
    T mide cuánto cambia la polarización con la frecuencia; la diferencia de fase
    entre sus autovalores, por unidad de frecuencia, es el DGD.

    - tau: DGD por sección, en segundos.
    - thetas, phis: ángulos de cada sección (largo K).
    - return: DGD total, en segundos.
    """
    k = len(thetas)
    if tau == 0 or k == 0:
        return 0.0
    dw = 1e-3 / (k * tau)  # dw · DGD << 1: así arg() no da la vuelta
    T = jones_matrix(dw, tau, thetas, phis) @ jones_matrix(0.0, tau, thetas, phis).conj().T
    rho = np.linalg.eigvals(T)
    return float(abs(np.angle(rho[0] / rho[1])) / dw)
