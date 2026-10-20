"""Bloque 07 - Error de temporización (Nivel 1)  [PENDIENTE]

En simple: el reloj de muestreo del receptor no cae en el instante óptimo, sino un
poco antes o después. Equivale a remuestrear la señal con un retardo fraccionario
(una fracción de la separación entre muestras).

Modelo:  y[n] = x(n - mu)   (valor entre dos muestras, obtenido por interpolación)
  El filtro ideal de retardo fraccionario es H(w) = exp(-j*w*mu), que no se puede
  implementar exacto; se lo aproxima con un interpolador.

Parámetros:
  mu : retardo fraccionario, en fracción de muestra (0 <= mu < 1). Por defecto 0.

Orden de trabajo sugerido:
  1) Interpolador lineal:  y[n] = (1 - mu) * x[n] + mu * x[n-1].
  2) Mejorarlo después (por ejemplo, un interpolador cúbico).

TODO:
  - Implementar el interpolador lineal en ``process``.
  - Comparar contra un retardo "ideal" hecho con FFT (multiplicar por exp(-j*w*mu)).
  - Definir qué se hace con la primera muestra, que no tiene una previa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import Block
from ..timing import cubic_delay, ideal_delay, linear_delay

METHODS = {"ideal": ideal_delay, "linear": linear_delay, "cubic": cubic_delay}


@dataclass
class TimingError(Block):
    mu: float = 0.0
    method: str = "ideal"
    name = "timing_error"

    def process(self, x: np.ndarray) -> np.ndarray:
        """Aplica el retardo fraccionario mu con el método elegido.

        Args:
            x: Señal de entrada, el tiempo es el último eje.

        Returns:
            Señal retardada, misma forma que x.
        """
        if self.method not in METHODS:
            raise ValueError(f"timing_error: método {self.method!r} desconocido (usar {sorted(METHODS)})")
        return METHODS[self.method](x, self.mu)
