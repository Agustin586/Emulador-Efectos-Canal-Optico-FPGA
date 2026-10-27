# Contexto para elegir la arquitectura de la FFT del bloque de CD

> Documento para pegar (o adjuntar) al empezar una conversación con Claude cuyo objetivo es **entender y elegir la arquitectura de la FFT** del bloque de dispersión cromática del emulador. Resume lo que ya está decidido y medido en el proyecto, para no rederivarlo. Fecha: 2026-10-27.

---

## 1. Pedido

Soy Wanda Molina y trabajo en el golden model del bloque de dispersión cromática (CD) de un emulador de canal óptico en FPGA (proyecto PROCOM, Fundación Fulgor). El algoritmo del bloque ya está cerrado y validado en punto flotante; ahora tengo que elegir **cómo se implementa la FFT en hardware**, porque el modelo en punto fijo bit a bit tiene que replicar exactamente su estructura (etapas, redondeos y escalado).

La referencia principal es **Fiber-on-Chip** (Börjeson y Larsson-Edefors, 2023): empezá por cómo implementan ahí la CD y la FFT, y compará el resto contra eso.

Quiero que me ayudes en tres pasos, y que me muestres el resultado de cada uno antes de seguir:

1. **Explicación.** Cómo funciona una FFT segmentada en hardware: radix-2, radix-4 y radix-2², DIT frente a DIF, arquitecturas SDF y MDC (y paralelas de P muestras por ciclo), orden de salida (natural o bit invertido), dónde están los multiplicadores y los twiddles, latencia y memoria. Con diagramas en texto para L = 16 y ejemplos numéricos chicos. Supone que conozco DSP en general pero no arquitecturas de FFT en FPGA.
2. **Opciones para este bloque.** Compará al menos: (0) la arquitectura de Fiber-on-Chip; (a) IP FFT de AMD/Xilinx (LogiCORE, PG109) en modo Pipelined Streaming I/O; (b) radix-2² SDF propia con P = 1; (c) radix-2² MDC o paralela con P > 1; (d) cualquier otra que te parezca razonable para overlap-save (por ejemplo, aprovechar el orden bit invertido para multiplicar por H̄ sin reordenar). Para cada una: recursos (multiplicadores, DSP, memoria), tasa, latencia, si permite cambiar N en tiempo de ejecución, y **qué hace falta para replicarla bit a bit** en el modelo de punto fijo.
3. **Recomendación y especificación.** Una arquitectura recomendada, parametrizada en N y P, con lo que el modelo de punto fijo necesita saber: descomposición y orden de las etapas, ancho de palabra de datos y twiddles por etapa, dónde se redondea y con qué modo, esquema de escalado (fijo por etapa, configurable, bloque flotante), orden de entrada y salida, latencia en ciclos. Y la lista de preguntas que tengo que llevarle al equipo.

### Reglas

- Distinguí siempre entre lo que sale de una fuente (citá documento y sección, por ejemplo PG109) y lo que es tu estimación o suposición. Si no sabés un dato de una IP o de un dispositivo, decilo; no lo inventes.
- Respondé en castellano.
- No escribas código de implementación todavía; pseudocódigo o Python corto para explicar está bien.

---

## 2. El algoritmo (cerrado, no se discute)

Overlap-save con muestreo en frecuencia (PDF del proyecto, §4.7.2; Jiang et al. 2026):

- La fibra es un pasa-todo: H(f) = exp(+jπDλ²z f²/c), más un término opcional de β₃. **Convención de signo positiva**, ya acordada con el grupo que hace el compensador (ellos aplican H\*).
- FFT de **L = 2N** puntos. Por bloque: ventana de [N muestras anteriores | N nuevas] → FFT → × H̄[k] → IFFT → se descartan las primeras N y se entregan las N últimas.
- **H̄[k] = H(f_k) · exp(−j2πk(N/2)/L)**: lo calcula el host una vez por configuración (|H̄| = 1). La fase lineal centra el chirp; la salida sale **retrasada N/2 muestras**.
- **N ≥ R**, con R = 2⌊|D|λ²z / (2cT_s²)⌋ + 1 la longitud del chirp en muestras. N = max(128, menor potencia de 2 ≥ R), hasta N_max. Entonces **L ∈ {256, 512, 1024, 2048, 4096}** y tiene que poder **cambiar con la distancia** (idealmente en tiempo de ejecución).
- Por bloque hay **una FFT y una IFFT**; el producto por H̄ es complejo por complejo.
- Entrada compleja (I y Q), una muestra compleja por instante de muestreo.

Código de referencia en Python (es el que ya está validado; `emulador/cd/overlap_save.py`):

```python
def cd_hbar(N, fs, D, lam, z, S=0.0):
    L = 2 * N
    k = np.fft.fftfreq(L) * L
    return cd_transfer(L, fs, D, lam, z, S) * np.exp(-2j * np.pi * k * (N / 2) / L)

def cd_overlap_save(x, Hbar):
    L = Hbar.shape[-1]; N = L // 2
    nb = -(-len(x) // N)
    xz = np.pad(x, (N, nb * N - len(x)))
    frames = xz[np.arange(nb)[:, None] * N + np.arange(L)]          # (nb, L)
    y = np.fft.ifft(np.fft.fft(frames, axis=-1) * Hbar, axis=-1)[:, N:]
    return y.reshape(-1)                                            # retrasada N/2
```

Longitud del chirp R (D = 17 ps/(nm·km), λ = 1550 nm), en muestras:

| f_s | 100 km | 200 km | 500 km | 1000 km | 2000 km |
|---|---|---|---|---|---|
| 64 GSa/s (32 GBd, 2 sps; escenario por defecto) | 55 | 111 | 279 | 559 | 1117 |
| 72 GSa/s (60 GBd, 1,2 sps; punto de diseño del compensador) | 71 | 141 | 353 | 707 | 1413 |
| 120 GSa/s (100 GBd, 1,2 sps) | 197 | 393 | 981 | 1961 | 3923 |

Con N_max = 2048 (provisorio), 100 GBd y 2000 km no entran.

## 3. Precisión del método en flotante (presupuesto de error)

El error de cuantización tiene que quedar por debajo del error propio del método, que ya está medido (NRMSE contra una FFT de toda la señal):

- 16-QAM, 2 sps: NRMSE entre 2·10⁻⁴ y 1,6·10⁻³ de 100 a 2000 km; peor caso dentro de N = 128: 9,8·10⁻³ (criterio < 1,5·10⁻²).
- 72 GSa/s: < 7,5·10⁻³ entre 50 y 2000 km; 1,8·10⁻³ en el punto de diseño (200 km, N = 256).
- 1,2 sps tiene más energía cerca de Nyquist: a 32 GBd y 600 km (R = 121, N = 128) llega a 1,1·10⁻².

Punto de partida en punto fijo de Jiang et al. (2026): señal en Q2.8 (10 bits) y H̄ de 10 bits, NRMSE ≈ 2·10⁻³ hasta 400 km.

## 4. Hardware (sin confirmar: todo parametrizado)

- Dispositivo: **RFSoC Zynq UltraScale+**, modelo **no confirmado** por el equipo. Según DS889, la familia tiene entre 1 200 y 4 272 bloques DSP. Por eso P, f_clk y N_max son parámetros (`HardwareParams` en `emulador/config.py`: P = 1, f_clk = None, N_max = 2048, provisorios).
- Tasa efectiva con overlap-save: **R_CD = P·f_clk·N/L = P·f_clk/2** (cada bloque de L muestras produce N nuevas). Ejemplo: para 64 GSa/s en tiempo real con f_clk = 500 MHz haría falta P = 256.
- Referencia: la versión **totalmente paralela** de Jiang et al. (núcleos Spiral radix-4) ocupa unos 4 370 DSP (63,9 % de una XCVU9P) **solo para la CD**: más que el RFSoC más grande.
- **Tasa: no definida (2026-10-27).** El equipo pide que **todo quede parametrizado** (f_s, P, f_clk, N_max); no hay un escenario objetivo. Las pruebas usan valores de interés (la tabla de R de §2). Quiero que la arquitectura sirva para cualquier tasa con solo cambiar P, que me muestres cómo escalan recursos y tasa con P, y si el resultado bit a bit puede ser idéntico para cualquier P (mismos redondeos y escalado).

Propuesta previa del proyecto (PDF §4.8, sin cerrar): algoritmo radix-2² (He y Torkelson, 1996), con la eficiencia en multiplicadores de radix-4 y cualquier L potencia de 2; arquitectura segmentada con P parametrizable (SDF con P = 1 en el prototipo, MDC con P > 1 si hace falta); prototipo con la IP FFT de Xilinx en Pipelined Streaming I/O (no Radix-4 Burst I/O, que procesa por ráfagas). Costo estimado: radix-2² SDF de 256 puntos ≈ 3 multiplicadores complejos (~10 DSP); FFT + × H̄ + IFFT ≈ 25–30 DSP. Revisá estas cifras.

## 5. Cómo se va a modelar el punto fijo

- Herramienta: **HALCON** (simulador de DSP en C++ de Fundación Fulgor). Tipo de punto fijo: `ac_fixed<W, I, S, Q, O>` de Algorithmic C (W bits totales, I enteros con el signo; redondeos AC_TRN, AC_RND, AC_RND_CONV…; desbordes AC_WRAP, AC_SAT…). Los formatos son parámetros de template; I y Q se manejan por separado.
- Además habrá un modelo rápido en Python (`emulador/quantization.py`) para explorar formatos sin recompilar, y que sirve de referencia bit a bit para HALCON.
- Por eso necesito que la especificación de la arquitectura diga exactamente **en qué punto de cada etapa se redondea o se escala, con qué ancho y qué modo**. Si recomendás la IP de Xilinx, decime qué documenta PG109 sobre su aritmética interna (escalado por etapa, redondeo convergente o truncado, ancho de twiddles) y si existe un modelo bit-accurate en C/MATLAB que se pueda usar como referencia.
- Las salidas que se exportan para verificar el RTL son las del hardware (retrasadas, sin alinear).

## 6. Preguntas que quiero tener respondidas al final

1. ¿IP de Xilinx o FFT propia? Ventajas para el modelo bit a bit de cada una.
2. ¿Radix-2² o radix-2 con P > 1? ¿Hay algo mejor para L entre 256 y 4096 con N variable?
3. ¿Cómo se cambia N en tiempo de ejecución sin resintetizar? ¿Qué implica para el modelo?
4. ¿Hace falta reordenar la salida bit invertida, o se puede multiplicar por H̄ en ese orden (cargando H̄ ya permutado) y alimentar la IFFT en ese orden?
5. ¿Qué esquema de escalado conviene, sabiendo que |H̄| = 1 y que la potencia de la señal se conserva?
6. Con overlap-save la FFT ve cada muestra dos veces (ventanas solapadas al 50 %): ¿cómo se organiza el buffer de entrada y cuánta memoria hace falta en función de N y P?
7. ¿Qué hace Fiber-on-Chip para emular a tasa reducida (tasa escalada, reloj, paralelismo) y qué conviene copiar?
8. Recursos y tasa en función de P para al menos dos RFSoC de la familia (el más chico y el más grande), aclarando qué es estimación.
9. ¿Qué parámetros nuevos tendría que agregar a `HardwareParams`?

## 7. Referencias del proyecto

- J. Jiang, C. Xie y M. Chen, "Hardware implementation of configurable chromatic dispersion model for real-time optical fiber communication simulation systems", *J. Lightwave Technol.*, 44(12), 2026.
- **(Principal)** E. Börjeson y P. Larsson-Edefors, "Fiber-on-Chip: Digital emulation of channel impairments for real-time DSP evaluation", *J. Lightwave Technol.*, 41(3), 2023.
- S. He y M. Torkelson, "A new approach to pipeline FFT processor", *Proc. IPPS*, 1996.
- AMD, *Zynq UltraScale+ RFSoC Data Sheet: Overview* (DS889).
- AMD, *Fast Fourier Transform LogiCORE IP Product Guide* (PG109).
- Bae y Gustafsson (2025), citado en el PDF del proyecto como ejemplo de FFT paralela segmentada (MDC).
