"""Métricas comunes del golden model de ISI y CD.

Todos los tests y notebooks usan estas funciones para medir. Cada una se valida
contra casos analíticos (tests/test_metrics.py): si una métrica estuviera mal,
todos los criterios que dependen de ella quedarían mal sin que nadie lo note.

Índice (qué métrica usar según qué se quiere saber):

  Pregunta                                          Métrica
  ------------------------------------------------  ---------------------------
  ¿Dos modelos dan la misma salida?                 nrmse
  ¿Cuánto se deformó la constelación?               evm_opt, snr_eq_db
  ¿Cuánto se ensanchó un pulso?                     rms_width
  ¿Cuánto tarda cada frecuencia en atravesar        group_delay
   un filtro?
  ¿Cuánta ISI deja un filtro, sin simular datos?    symbol_rate_response +
                                                    isi_power_ratio /
                                                    peak_distortion

Convención de frecuencias (GUIA_ISI_CD.md §1 y referencia §3): los espectros
entran en el orden de numpy.fft (fftfreq); fftshift se aplica solo adentro, para
derivar o graficar.

Los ejemplos de cada docstring se verifican con:
    pytest --doctest-modules emulador/metrics.py
"""
from __future__ import annotations

import numpy as np

from .common import rrc_taps


# --------------------------------------------------------------------------
# Comparación entre modelos
# --------------------------------------------------------------------------
def nrmse(y: np.ndarray, ref: np.ndarray) -> float:
    """Error cuadrático medio normalizado entre una salida y su referencia.

    QUÉ MIDE
        Cuánto se aparta ``y`` de ``ref``, como fracción de la energía de
        ``ref``. Es la métrica para comparar dos modelos del mismo bloque
        (referencia §1, PDF Ec. 32):

            NRMSE = sqrt( sum_n |y[n] - ref[n]|^2 / sum_n |ref[n]|^2 )

    CÓMO INTERPRETARLA
        Es adimensional y no depende de la potencia de la señal. Órdenes de
        magnitud en el proyecto:
          ~1e-15 a 1e-12 : iguales salvo redondeo de punto flotante
                           (identidad con z = 0, reversibilidad de la CD,
                           composición de dos tramos).
          < 1e-2         : overlap-save contra la referencia con N >= R
                           (error del método, PDF Fig. 9).
          ~2e-3          : cuantización de la CD con 10 bits (Jiang et al.).
          > 1e-1         : el modelo está mal o falta alinear.
        Se puede leer como una "SNR del modelo": -20·log10(NRMSE) dB
        (1e-2 equivale a 40 dB).

    DÓNDE SE USA
        Separar los errores de cada nivel del golden model: referencia ideal
        contra modelo por bloques (error del método), y modelo flotante contra
        punto fijo (error de cuantización). Pasos B2, B3, B4 y C3.

    PRECAUCIONES
        - ``y`` y ``ref`` tienen que tener la misma forma y estar YA alineados:
          hay que descontar el retardo de cada modelo (por ejemplo, N/2 muestras
          del overlap-save) antes de llamar. Si no se alinean, el NRMSE da
          cercano a 1 o más, aunque los dos modelos sean correctos.
        - Hay que descartar los transitorios de los bordes si un modelo los
          tiene y el otro no.
        - Acepta (n,) o (2, n); con dos polarizaciones suma sobre ambas.

    Ejemplo:
        >>> x = np.array([1 + 1j, 2 - 1j, -1j])
        >>> round(nrmse(1.01 * x, x), 12)
        0.01
    """
    y = np.asarray(y)
    ref = np.asarray(ref)
    if y.shape != ref.shape:
        raise ValueError(f"nrmse: formas distintas {y.shape} y {ref.shape} (¿faltó alinear?)")
    energy_ref = np.sum(np.abs(ref) ** 2)
    if energy_ref == 0:
        raise ValueError("nrmse: la referencia tiene energía nula")
    return float(np.sqrt(np.sum(np.abs(y - ref) ** 2) / energy_ref))


# --------------------------------------------------------------------------
# Calidad de la constelación
# --------------------------------------------------------------------------
def evm_opt(rx: np.ndarray, ref: np.ndarray, discard: int = 0) -> float:
    """EVM (magnitud del vector de error) con ganancia y fase óptimas.

    QUÉ MIDE
        Cuánto se apartan los símbolos recibidos ``rx`` de los transmitidos
        ``ref``, después de corregir la mejor ganancia compleja constante
        (amplitud y rotación):

            g   = sum conj(rx)·ref / sum |rx|^2    (ganancia de mínimo error)
            EVM = ||g·rx - ref|| / ||ref||

        Devuelve una FRACCIÓN (0.05 = 5 %), no un porcentaje.

    POR QUÉ CORREGIR GANANCIA Y FASE
        Una ganancia o una rotación constantes no son distorsión: cualquier
        receptor las corrige con un escalar. Lo que queda después de corregirlas
        es lo que realmente degrada la detección (ISI, ruido, residuo de beta3).
        Es indispensable para la CD: su cursor principal es complejo y rota
        toda la constelación, así que sin corregir la fase la EVM mediría esa
        rotación y no la dispersión.

    DIFERENCIA CON common.evm_percent
        ``evm_percent`` normaliza solo la potencia y NO corrige la fase: una
        rotación cuenta como error. Sirve para ver el efecto de un bloque de
        offset de fase. Las dos se usan, según qué se quiera medir.

    CÓMO INTERPRETARLA
        Con un error e ortogonal a ref (ruido, ISI no correlacionada), la
        ganancia óptima absorbe parte de la potencia y resulta
        EVM^2 = N/(S+N), no N/S. A SNR alta la diferencia es despreciable.
        Referencias del proyecto (QAM16, PDF Tabla 17, residuo de beta3 tras
        1000 km): 0.7 % a 32 GBd (despreciable), 5.6 % a 64 GBd, 22 % a
        100 GBd, > 100 % a 200 GBd (constelación destruida). Piso del RRC
        truncado sin canal: ~0.1-0.2 %.

    DÓNDE SE USA
        Constelaciones de los notebooks, receptor mínimo (paso P4), residuo de
        beta3 (paso B4).

    PARÁMETROS
        discard : símbolos que se descartan de cada borde (transitorios del
                  filtrado). Usar al menos el span del RRC.

    Ejemplo (una rotación y una ganancia no cuentan como error):
        >>> ref = np.array([1, -1, 1j, -1j])
        >>> evm_opt(2 * np.exp(0.5j) * ref, ref) < 1e-12
        True
    """
    rx = np.asarray(rx)
    ref = np.asarray(ref)
    if rx.shape != ref.shape:
        raise ValueError(f"evm_opt: formas distintas {rx.shape} y {ref.shape}")
    if discard:
        rx = rx[..., discard:-discard]
        ref = ref[..., discard:-discard]
    energy_rx = np.sum(np.abs(rx) ** 2)
    if energy_rx == 0 or np.sum(np.abs(ref) ** 2) == 0:
        raise ValueError("evm_opt: rx o ref tienen energía nula")
    g = np.vdot(rx, ref) / energy_rx
    return float(np.linalg.norm(g * rx - ref) / np.linalg.norm(ref))


def snr_eq_db(rx: np.ndarray, ref: np.ndarray, discard: int = 0) -> float:
    """SNR equivalente en dB de una distorsión: -20·log10(evm_opt).

    QUÉ MIDE
        Expresa una distorsión determinista (ISI, residuo de beta3) en la misma
        escala que el ruido del enlace. Así se puede responder "¿esta ISI es
        más o menos grave que el ruido?" comparando dos números en dB
        (PDF Fig. 10).

    CÓMO INTERPRETARLA
        Mayor es mejor. Si la distorsión se comporta como un error ortogonal a
        la señal, vale 10·log10(1 + S/N), que a SNR alta es S/N. Referencias
        del proyecto (residuo de beta3 tras 1000 km, PDF Tabla 17):
          43 dB : despreciable frente al ruido del enlace (32 GBd);
          25 dB : comparable a una fuente de ruido más (64 GBd);
          13 dB : domina sobre el ruido si no se ecualiza (100 GBd);
          < 0 dB: constelación destruida (200 GBd).

    DÓNDE SE USA
        Residuo de beta3 (paso B4); SNR equivalente de la ISI medida sobre una
        señal (paso A5). Para la ISI de un filtro sin simular datos, usar
        ``isi_power_ratio`` sobre ``symbol_rate_response``, que es determinista.

    Devuelve inf si no hay error.

    Ejemplo (error ortogonal con N/S = 0.01 -> 10·log10(101)):
        >>> ref = np.ones(4)
        >>> round(snr_eq_db(ref + 0.1 * np.array([1, -1, 1, -1]), ref), 2)
        20.04
    """
    evm = evm_opt(rx, ref, discard)
    return float(np.inf) if evm == 0 else float(-20 * np.log10(evm))


# --------------------------------------------------------------------------
# Forma de pulso
# --------------------------------------------------------------------------
def rms_width(a: np.ndarray, t: np.ndarray) -> float:
    """Ancho rms (desvío estándar temporal) de un pulso, ponderado por |a|^2.

    QUÉ MIDE
        Cuán extendido está un pulso en el tiempo. Trata la intensidad |a(t)|^2
        como una distribución y calcula su desvío estándar:

            <t>   = sum t |a|^2 / sum |a|^2
            sigma = sqrt( sum (t - <t>)^2 |a|^2 / sum |a|^2 )

        Devuelve sigma en las mismas unidades que ``t`` (segundos).

    CÓMO INTERPRETARLA
        - No depende de la posición del pulso (un retardo no lo ensancha), ni
          de su amplitud, ni de su fase (el chirp de la CD cambia la fase, pero
          lo que se mide es la intensidad).
        - Para un pulso gaussiano a = exp(-t^2 / 2T0^2) vale T0/sqrt(2).
        - Bajo CD sin beta3, el pulso gaussiano sigue siendo gaussiano y su
          ancho crece como T(z) = T0·sqrt(1 + (z/L_D)^2), con L_D = T0^2/|beta2|
          (PDF Ec. 22-23). El ancho rms crece en el mismo factor. Ejemplo del
          PDF: T0 = 15.6 ps en SSMF da L_D = 11.2 km.

    DÓNDE SE USA
        Test del pulso gaussiano de la CD (paso B2) y su figura (PDF Fig. 5).

    PRECAUCIONES
        La grilla ``t`` tiene que cubrir todo el pulso (incluidas las colas del
        pulso ensanchado) y muestrearlo bien: con T0 cercano a Ts hay aliasing.
        Usar fs alta o T0 >= 3·Ts.

    Ejemplo:
        >>> T0 = 10e-12
        >>> t = np.linspace(-100e-12, 100e-12, 4001)
        >>> round(rms_width(np.exp(-t**2 / (2 * T0**2)), t) / T0, 4)
        0.7071
    """
    a = np.asarray(a)
    t = np.asarray(t, dtype=float)
    if a.shape != t.shape:
        raise ValueError(f"rms_width: formas distintas {a.shape} y {t.shape}")
    w = np.abs(a) ** 2
    energy = np.sum(w)
    if energy == 0:
        raise ValueError("rms_width: pulso de energía nula")
    mean = np.sum(t * w) / energy
    return float(np.sqrt(np.sum((t - mean) ** 2 * w) / energy))


# --------------------------------------------------------------------------
# Respuesta en frecuencia
# --------------------------------------------------------------------------
def group_delay(H: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Retardo de grupo tau(f) de un filtro, medido sobre su respuesta en frecuencia.

    QUÉ MIDE
        Cuánto tarda en atravesar el filtro la energía en torno a cada
        frecuencia f (PDF Ec. 5):

            tau(f) = -(1/2pi)·dphi/df = -dphi/domega,   H(f) = |H(f)|·exp(j·phi(f))

        Es una MEDICIÓN numérica sobre una H ya calculada. No usa parámetros
        físicos: D, lambda, z o los coeficientes del filtro ya están dentro de
        los valores de H. ``fs`` solo sirve para saber a qué frecuencia
        corresponde cada muestra de H (f_k = k·fs/L).

    CÓMO INTERPRETARLA
        - tau constante: el filtro solo retrasa, no deforma la señal (fase
          lineal: retardo puro, FIR simétrico, gaussiano, Bessel en su banda).
        - tau variable dentro de la banda: el filtro deforma la señal y produce
          ISI por fase (Butterworth, CD).
        - Para la CD sin beta3, tau(f) = -D·lambda^2·z·f/c es una recta de
          pendiente negativa (convención de signo positiva, GUIA §1). Con SSMF,
          100 km y 1550 nm, la pendiente es -13.6 ps/GHz y el retardo en el
          borde de una banda de 64 GSa/s es de unos 436 ps. Con beta3, tau(f)
          es una parábola.

    DÓNDE SE USA
        Verificar que un filtro implementado coincide con la teoría:
        - CD (paso B1): la pendiente medida sobre H = cd_transfer(...) tiene que
          ser -D·lambda^2·z/c; detecta el factor 1/2 de la guía original y un
          signo invertido.
        - ISI (paso A3): Bessel y Butterworth truncados a K taps, que no tienen
          fórmula cerrada, contra su prototipo analógico (< 0.4 muestras).

    PARÁMETROS
        H  : respuesta en la grilla de una FFT de L puntos, en orden numpy.fft
             (f = fftfreq(L, 1/fs)). Por ejemplo, np.fft.fft(h, L) para un FIR.
        fs : frecuencia de muestreo [Hz].

    DEVUELVE
        (f, tau): f ordenada de -fs/2 a fs/2 [Hz] y tau [s]. Para expresar tau
        en muestras, multiplicar por fs.

    PRECAUCIONES
        - La fase se desenrolla (unwrap) antes de derivar: sin eso, los saltos
          de 2*pi aparecen como picos de retardo. El unwrap supone que la fase
          cambia menos de pi entre muestras de frecuencia; si no se cumple
          (retardos de más de L/2 muestras), usar una FFT más larga.
        - La derivada usa diferencias de segundo orden también en los bordes
          (edge_order=2), así es exacta para fases lineales y cuadráticas. Con
          el valor por defecto de np.gradient, el borde de una fase cuadrática
          tendría un error relativo de ~5e-4.
        - Donde |H| es casi cero la fase no está definida y tau no tiene
          sentido: mirar solo la banda donde el filtro no atenúa demasiado.

    Ejemplo (retardo puro de 3 muestras):
        >>> fs = 64e9
        >>> f = np.fft.fftfreq(64, 1 / fs)
        >>> _, tau = group_delay(np.exp(-2j * np.pi * f * 3 / fs), fs)
        >>> bool(np.allclose(tau * fs, 3))
        True
    """
    H = np.asarray(H) # Asegura que sea un array de numpy
    L = H.shape[-1]   # Número de puntos de la FFT
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1 / fs)) # Frecuencias ordenadas de -fs/2 a fs/2
    phase = np.unwrap(np.angle(np.fft.fftshift(H))) # Desenrolla la fase
    # Calcula el retardo de grupo como la derivada negativa de la fase respecto a la frecuencia angular
    tau = -np.gradient(phase, 2 * np.pi * f, edge_order=2)
    return f, tau


# --------------------------------------------------------------------------
# Severidad de la ISI (PDF §3.7)
# --------------------------------------------------------------------------
def _split_cursor(p: np.ndarray) -> tuple[float, np.ndarray]:
    """Separa el cursor principal (el de mayor módulo) del resto de la respuesta."""
    p = np.asarray(p)
    if not np.any(p):
        raise ValueError("respuesta nula: no hay cursor principal")
    n0 = int(np.argmax(np.abs(p)))
    return float(np.abs(p[n0])), np.delete(p, n0)


def peak_distortion(p: np.ndarray) -> float:
    """Distorsión de pico D_p de una respuesta muestreada a la tasa de símbolos.

    QUÉ MIDE
        La peor ISI posible: el caso en que todos los símbolos vecinos suman su
        interferencia en contra del símbolo deseado (PDF Ec. 13):

            D_p = sum_{n != n0} |p[n]| / |p[n0]|

        ``p`` es la respuesta total (transmisor + canal + filtro adaptado)
        muestreada a la tasa de símbolos: p[n0] es el cursor principal (el de
        mayor módulo, aunque sea negativo o complejo) y el resto son los
        pre-cursores (n < n0) y post-cursores (n > n0). Obtenerla con
        ``symbol_rate_response``.

    CÓMO INTERPRETARLA
        - D_p = 0: sin ISI (criterio de Nyquist cumplido).
        - D_p < 1: el ojo sigue abierto para modulaciones de dos niveles; el
          peor caso de ISI no alcanza para cruzar el umbral de decisión.
        - D_p >= 1: el ojo se cierra en el peor caso. Con más niveles (PAM4,
          QAM16) el umbral está más cerca y el ojo se cierra antes.
        Es una cota pesimista: para una medida típica, usar isi_power_ratio.

    DÓNDE SE USA
        Magnitud derivada que se reporta con cada configuración de la ISI
        (PDF §6.3, paso A5).

    Ejemplo (cursor 1 con ecos 0.1 y -0.2):
        >>> round(peak_distortion(np.array([0.1, 1.0, -0.2])), 6)
        0.3
    """
    cursor, rest = _split_cursor(p)
    return float(np.sum(np.abs(rest)) / cursor)


def isi_power_ratio(p: np.ndarray) -> float:
    """Potencia de ISI relativa al cursor principal.

    QUÉ MIDE
        La ISI típica, en potencia: cuánta potencia aportan los símbolos
        vecinos respecto de la del símbolo deseado, suponiendo símbolos
        independientes y de igual potencia:

            ISI = sum_{n != n0} |p[n]|^2 / |p[n0]|^2

        Con ``p`` como en ``peak_distortion`` (respuesta a la tasa de símbolos,
        cursor = el de mayor módulo).

    CÓMO INTERPRETARLA
        Su inversa en dB, -10·log10(ISI), es la "SNR equivalente" de la ISI:
        se compara directamente con la SNR del enlace (mayor es mejor).
        Valores del PDF (Tabla 4, instante óptimo, roll-off 0.1): con
        B = 0.6·Rs, 23.0 dB (gaussiano), 23.3 dB (Bessel m = 4) y 25.9 dB
        (Butterworth m = 4); con B = 0.4·Rs, entre 12.2 y 16.2 dB. Piso del
        RRC truncado sin canal: -44.6 dB (roll-off 0.1, span 32).

    DÓNDE SE USA
        Reproducir la Tabla 4 del PDF (paso A5) y cuantificar la ISI de cada
        perfil de forma determinista, sin simular símbolos aleatorios.

    Ejemplo (ecos 0.1 y -0.2 -> 0.01 + 0.04; SNR equivalente de 13 dB):
        >>> r = isi_power_ratio(np.array([0.1, 1.0, -0.2]))
        >>> round(r, 6), round(float(-10 * np.log10(r)), 1)
        (0.05, 13.0)
    """
    cursor, rest = _split_cursor(p)
    return float(np.sum(np.abs(rest) ** 2) / cursor**2)


def symbol_rate_response(
    h: np.ndarray, sps: int, rolloff: float, span: int = 32, phase: int = 0
) -> np.ndarray:
    """Respuesta total de la cadena, muestreada a la tasa de símbolos.

    QUÉ CALCULA
        Lo que "ve" el decisor del receptor cuando se transmite un único
        símbolo: el pulso del transmisor (RRC), el canal ``h`` y el filtro
        adaptado del receptor (RRC), tomando una muestra por símbolo:

            p[n] = (RRC * h * RRC)[phase + n·sps]

        Sin canal, RRC * RRC es un coseno realzado, que cumple Nyquist: vale 1
        en el pico (``rrc_taps`` tiene energía unitaria) y cero en los demás
        instantes de símbolo. Cualquier ``h`` que no sea un retardo puro rompe
        esa condición y produce ISI (PDF §3.2).

    PARA QUÉ SIRVE
        Es la p[n] que necesitan ``peak_distortion`` e ``isi_power_ratio``.
        Permite medir la ISI de un filtro de forma exacta y rápida, sin generar
        símbolos aleatorios ni pasar por el receptor completo.

    PARÁMETROS
        h       : coeficientes del canal a la frecuencia de muestreo (no a la
                  tasa de símbolos): con sps = 2, un eco de un símbolo está 2
                  coeficientes más adelante.
        sps     : muestras por símbolo.
        rolloff : roll-off del RRC (0.1 en el escenario de referencia).
        span    : largo del RRC en símbolos.
        phase   : fase de muestreo, 0 <= phase < sps. Selecciona en qué muestra
                  de cada símbolo se decide. Sin canal, phase = 0 cae en el pico.

    PRECAUCIONES
        - El RRC truncado deja un piso de ISI propio: con roll-off 0.1 es de
          -44.6 dB con span = 32 y de -35.5 dB con span = 16. Por eso span = 32
          por defecto; con span menor, el piso tapa ISI chicas.
        - Los perfiles causales (Bessel, Butterworth) agregan un retardo con
          parte fraccionaria. Con la fase equivocada la ISI medida puede ser
          más de 10 dB peor que la real (PDF §3.5.5). Para el instante óptimo,
          probar todas las fases o usar un escenario con más sps (por ejemplo,
          sps = 16).

    Ejemplo (eco de un símbolo con amplitud 0.3 a 2 sps -> ISI ~ 0.3^2):
        >>> p = symbol_rate_response([1.0, 0.0, 0.3], sps=2, rolloff=0.1)
        >>> round(isi_power_ratio(p), 2)
        0.09
    """
    if not 0 <= phase < sps:
        raise ValueError(f"phase debe estar entre 0 y sps - 1 = {sps - 1}: {phase!r}")
    g = rrc_taps(rolloff, span, sps)
    total = np.convolve(np.convolve(g, np.asarray(h)), g)
    return total[phase::sps]
