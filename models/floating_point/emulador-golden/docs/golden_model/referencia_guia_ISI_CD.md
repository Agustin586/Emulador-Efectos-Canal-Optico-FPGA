# Golden model de ISI genérica y dispersión cromática

**Guía de desarrollo** — PROCOM · Fundación Fulgor

Guía personal para desarrollar, en Python, el modelo de referencia de mis dos bloques del emulador de canal óptico: **ISI genérica** y **dispersión cromática (CD)**. Reúne las decisiones tomadas, las ecuaciones de la bibliografía, el orden de trabajo, los tests con sus criterios y la forma de documentar los resultados en notebooks.

---

## Índice

- [0. Resumen de decisiones](#0-resumen-de-decisiones)
- [1. Qué es el golden model](#1-qué-es-el-golden-model)
- [2. Escenario de referencia configurable](#2-escenario-de-referencia-configurable)
- [3. Convenciones](#3-convenciones)
- [4. Estructura del repositorio](#4-estructura-del-repositorio)
- [5. Entorno](#5-entorno)
- [6. Plan de trabajo](#6-plan-de-trabajo)
- [Parte A · ISI genérica](#parte-a--isi-genérica)
- [Parte B · Dispersión cromática](#parte-b--dispersión-cromática)
- [Parte C · Etapas comunes](#parte-c--etapas-comunes)
- [7. Notebooks](#7-notebooks)
- [8. Checklist](#8-checklist)
- [9. Referencias](#9-referencias)

---

## 0. Resumen de decisiones

| Tema | Decisión | Estado |
|---|---|---|
| Escenario de referencia | 32 GBd, 2 muestras por símbolo, f<sub>s</sub> = 64 GSa/s, **configurable** (sección 2) | Cerrado |
| Convención de signo de la CD | **Positiva**: H<sub>CD</sub>(f) = exp(+jπDλ²z f²/c) (sección 3) | Cerrado |
| Factor ½ de la guía del proyecto | **Descartado**: no se usa; la ecuación correcta es la de B.1 | Cerrado |
| Término β₃ | Incluido; pendiente S = 0 por defecto | Cerrado |
| Método de la CD | Muestreo en frecuencia + overlap-save, FFT de L = 2N, regla N ≥ R | Cerrado |
| Diseño de coeficientes de la ISI | En frecuencia; Bessel y Butterworth desde el prototipo analógico | Cerrado |
| Perfil principal de la ISI | Bessel, m = 4, B = 0,6 R<sub>s</sub>, K = 16 | Cerrado |
| Plataforma de hardware | **RFSoC** (Zynq UltraScale+); modelo exacto sin confirmar | Parcial |
| FFT del hardware | Radix-2², paralelismo P parametrizable; P se fija con el dispositivo | Parcial |
| Formatos de punto fijo | A definir en la etapa de cuantización | Pendiente |

---

## 1. Qué es el golden model

Es la **referencia de verdad** contra la que se compara el RTL. Cada bloque se implementa en tres niveles, y separarlos permite saber de dónde viene cada error:

| Nivel | Qué es | Qué mide |
|---|---|---|
| **Referencia ideal** | La física calculada con la mayor exactitud posible; no es implementable en hardware. | Define "la verdad". |
| **Modelo por bloques (flotante)** | El algoritmo que va a ejecutar el RTL, en `float64`. | El **error del método**. |
| **Modelo por bloques (punto fijo)** | Lo anterior con la cuantización y la estructura del hardware. | El **error de cuantización**; produce los vectores esperados bit a bit. |

Métrica común entre niveles, con las salidas alineadas (descontando retardos fijos):

$$
\mathrm{NRMSE} = \sqrt{\frac{\sum_n |y[n]-y_{\mathrm{ref}}[n]|^2}{\sum_n |y_{\mathrm{ref}}[n]|^2}}
$$

En la ISI, el FIR del hardware y la referencia coinciden: la ISI no tiene error de método y solo aparece el de cuantización. En la CD los tres niveles son distintos.

---

## 2. Escenario de referencia configurable

Todo lo que depende de la velocidad del sistema (f<sub>s</sub>, T<sub>s</sub>, B de los perfiles, R de la CD, N) se deriva de un único objeto `Escenario`. Cambiar de escenario no requiere tocar ningún otro módulo.

| Parámetro | Por defecto | Rango aceptado | Comentario |
|---|---|---|---|
| R<sub>s</sub> | 32 GBd | 1 a 300 GBd | El proyecto apunta a ≥ 100 GBd. |
| sps | 2 | 1, 2, 4, 8, 16 | 16 solo para análisis fino (instante óptimo de muestreo). |
| λ | 1550 nm | 1530 a 1565 nm | Banda C. |
| roll-off del RRC | 0,1 | 0,05 a 0,5 | |
| f<sub>s</sub> = R<sub>s</sub>·sps | 64 GSa/s | derivado | |

```python
# emulador/config.py
from dataclasses import dataclass

C = 299_792_458.0  # m/s

def D_si(D_ps_nm_km):
    """ps/(nm·km) -> s/m^2"""
    return D_ps_nm_km * 1e-12 / (1e-9 * 1e3)

def S_si(S_ps_nm2_km):
    """ps/(nm^2·km) -> s/m^3"""
    return S_ps_nm2_km * 1e-12 / (1e-18 * 1e3)

@dataclass(frozen=True)
class Escenario:
    """Escenario de simulación. Por defecto: 32 GBd, 2 sps -> fs = 64 GSa/s."""
    Rs: float = 32e9          # tasa de símbolos [Bd]
    sps: int = 2              # muestras por símbolo
    lam: float = 1550e-9      # longitud de onda [m]
    rolloff: float = 0.1      # roll-off del RRC

    def __post_init__(self):
        if not (1e9 <= self.Rs <= 300e9):
            raise ValueError("Rs fuera de rango (1 a 300 GBd)")
        if self.sps not in (1, 2, 4, 8, 16):
            raise ValueError("sps debe ser 1, 2, 4, 8 o 16")
        if not (1530e-9 <= self.lam <= 1565e-9):
            raise ValueError("lambda fuera de la banda C")

    @property
    def fs(self) -> float:
        return self.Rs * self.sps

    @property
    def Ts(self) -> float:
        return 1 / self.fs
```

Uso: `esc = Escenario()` para el de referencia, `Escenario(Rs=100e9)` para uno de alta velocidad. El escenario se guarda en los metadatos de cada simulación y de cada vector exportado.

---

## 3. Convenciones

| Aspecto | Convención |
|---|---|
| Unidades | SI en todo el código: D en s/m², S en s/m³, λ y z en m, β₂ en s²/m, β₃ en s³/m. Las unidades de hoja de datos se convierten solo en `config.py`. |
| Señal | Envolvente compleja en banda base, `complex128`, una polarización; f es la desviación respecto de la portadora, en [−f<sub>s</sub>/2, f<sub>s</sub>/2). |
| Transformada | La de `numpy.fft`: X[k] = Σ x[n]·e^(−j2πkn/L). |
| Eje de frecuencias | Siempre `np.fft.fftfreq(L, 1/fs)`; `fftshift` solo para graficar. |
| **Signo de la CD** | **Positivo**: H<sub>CD</sub>(f) = exp(+jπDλ²z f²/c), equivalente a exp(−jβ(ω)z) con β₂ = −Dλ²/(2πc). Con D > 0, las frecuencias altas llegan antes (retardo de grupo de pendiente negativa). |
| Potencia | Estímulos con potencia media unitaria. |
| Retardos | Cada bloque declara su retardo fijo en muestras (`bloque.retardo`). |
| Aleatoriedad | `np.random.default_rng(semilla)`; nunca el estado global de numpy. |

> La convención de signo queda **fijada por un test** (B.6): si alguien cambia el signo, el test falla. Hay que comunicarla a la tesis de compensación (CDC) para que su filtro use la conjugada.

Para integrarse con el resto del emulador, ambos bloques heredan de una clase `Bloque` mínima:

```python
# emulador/base.py
import numpy as np

class Bloque:
    """Un deterioro del canal. Parámetros en SI; señal compleja a fs."""
    def __init__(self, fs: float):
        self.fs = fs

    @property
    def retardo(self) -> int:
        """Retardo fijo que introduce el bloque, en muestras."""
        return 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def coeficientes(self) -> dict:
        """Lo que se cargaría en los registros o memorias del hardware."""
        return {}
```

---

## 4. Estructura del repositorio

```
emulador-golden/
├── pyproject.toml
├── README.md                    ← esta guía
├── emulador/                    ← toda la lógica vive acá
│   ├── __init__.py
│   ├── config.py                ← constantes, conversión de unidades, Escenario
│   ├── base.py                  ← clase Bloque
│   ├── estilo.mplstyle          ← estilo de gráficos (paleta de las presentaciones)
│   ├── estimulos/
│   │   ├── prbs.py              ← PRBS-15 / PRBS-31
│   │   ├── mapper.py            ← PAM4, QAM16 (Gray)
│   │   ├── pulso.py             ← RRC y sobremuestreo
│   │   └── pruebas.py           ← impulso, pulso gaussiano, tonos
│   ├── isi/
│   │   ├── perfiles.py          ← gaussiano, Bessel, Butterworth, eco, medido, CD corta
│   │   └── bloque.py            ← ParametrosISI, clase ISI(Bloque)
│   ├── cd/
│   │   ├── fisica.py            ← betas, cd_transfer, cd_R, cd_N
│   │   ├── referencia.py        ← cd_referencia (FFT completa)
│   │   ├── overlap_save.py      ← cd_hbar, cd_overlap_save
│   │   └── bloque.py            ← ParametrosCD, clase CD(Bloque)
│   ├── rx.py                    ← receptor mínimo (filtro adaptado, muestreo)
│   ├── metricas.py              ← NRMSE, EVM, SNR equivalente, retardo de grupo, ancho rms
│   ├── graficos.py              ← constelación, ojo, espectro, guardar figuras
│   ├── cuantizacion.py          ← Qm.n, redondeo, saturación
│   └── exportar.py              ← vectores .npy/.hex + meta.json
├── notebooks/                   ← teoría, gráficos y conclusiones (importan de emulador/)
├── scripts/                     ← barridos y exportación por línea de comandos
├── tests/
│   ├── test_config.py
│   ├── test_isi.py
│   └── test_cd.py
├── vectores/                    ← salida para el RTL (no se versiona)
└── docs/
    ├── figuras/
    └── referencias/
```

**Regla principal:** la lógica vive en `emulador/`; notebooks, scripts y tests solo la usan. Así lo que se grafica es exactamente lo que se testea y lo que genera los vectores del RTL.

---

## 5. Entorno

```toml
# pyproject.toml
[project]
name = "emulador"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "scipy>=1.11", "matplotlib>=3.8"]

[project.optional-dependencies]
dev = ["pytest>=8", "jupyterlab>=4", "ipykernel", "jupytext>=1.16", "nbstripout", "nbconvert"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```bash
pip install -e ".[dev]"      # modo editable: los notebooks ven los cambios sin reinstalar
nbstripout --install         # borra las salidas de los notebooks al commitear
pytest                       # corre todos los tests
```

En el contenedor Docker con code-server, abrir los notebooks con la extensión de Jupyter de VS Code o con `jupyter lab --no-browser`.

---

## 6. Plan de trabajo

Cada etapa termina con sus tests pasando y su notebook actualizado. **Los tests se escriben junto con cada módulo, no al final.**

| Etapa | Contenido | Módulos | Notebook | Listo cuando |
|---|---|---|---|---|
| 0 | Base: escenario, unidades, interfaz, métricas | `config`, `base`, `metricas` | 00 | `test_config` pasa |
| 1 | Estímulos | `estimulos/`, `rx` | 01 | Constelación y ojo limpios |
| 2 | ISI: bloque y perfiles | `isi/` | 02 | Tests de la tabla A.6 |
| 3 | CD: física y referencia | `cd/fisica`, `cd/referencia` | 03 | Tests de física de B.6 |
| 4 | CD: overlap-save | `cd/overlap_save`, `cd/bloque` | 04 | NRMSE < 10⁻² con N ≥ R |
| 5 | Cuantización | `cuantizacion` | 05 | Curvas NRMSE contra bits |
| 6 | Vectores para el RTL | `exportar` | 06 | Vectores + meta.json por caso |

Las etapas 0 y 1 son la base común (Parte C describe las que no son específicas de un bloque). Las etapas 2 y 3–4 son independientes entre sí.

---

## Parte A · ISI genérica

### A.1 Modelo

Filtro FIR complejo causal de K coeficientes, definidos a f<sub>s</sub>:

$$
y[n] = \sum_{k=0}^{K-1} h[k]\,x[n-k], \qquad
H_{\mathrm{ISI}}(f) = \sum_{k=0}^{K-1} h[k]\,e^{-j2\pi f k T_s}, \qquad
\sum_k |h[k]|^2 = 1
$$

- La normalización por energía evita que el bloque cambie la potencia de la señal (y la SNR que fije el bloque de ruido).
- El cursor principal es k₀ = argmax |h[k]|; su retardo no es ISI y se descuenta al comparar.
- Con 2 sps, un eco "de un símbolo" está 2 coeficientes más adelante.

Referencias: Proakis y Salehi (ISI, criterio de Nyquist); Proakis y Manolakis (filtros FIR).

### A.2 Perfiles de coeficientes

El FIR es la **estructura**; los perfiles son formas de **calcular** h[k]. En hardware solo cambia el vector cargado.

| Perfil | Fase | Uso |
|---|---|---|
| Bypass, h = [1] | Lineal | Verificar la cadena. |
| Eco, h = [1, a₁, a₂, …] | Según los ecos | Reflexiones; canal simple para ecualizadores. |
| Gaussiano | Lineal | **Validación** (fórmula cerrada, simétrico). |
| **Bessel, orden m** | Casi lineal | **Perfil principal**: componente real con retardo constante. |
| Butterworth, orden m | No lineal | Distorsión de fase y **estrés**. |
| Medido o simulado | La del componente | Respuestas del FI-DAC y drivers (tesis analógicas). |
| CD corta | Cuadrática | Taps de Savory para pocos km (vincula con la Parte B). |

El **orden m** es la cantidad de polos: fija la pendiente de caída (−20·m dB/década) y, en el Butterworth, la distorsión de fase y el sobrepico (el Bessel mantiene la fase lineal con cualquier m).

### A.3 Diseño de los coeficientes (en frecuencia)

**Gaussiano**, con H<sub>G</sub>(f) = exp(−(ln 2/2)·f²/B²), centrado antes de truncar (k₀ = ⌊K/2⌋):

```python
import numpy as np
from scipy import signal

def normalizar_energia(h):
    return h / np.sqrt(np.sum(np.abs(h) ** 2))

def perfil_gaussiano(B, fs, K, L=8192):
    """Pasa-bajos gaussiano con -3 dB en B, diseñado en frecuencia (centrado)."""
    f = np.fft.fftfreq(L, d=1 / fs)
    H = np.exp(-np.log(2) / 2 * f**2 / B**2)
    h = np.fft.fftshift(np.fft.ifft(H)).real
    c = L // 2
    return normalizar_energia(h[c - K // 2 : c - K // 2 + K])
```

**Bessel y Butterworth**, desde el prototipo analógico de orden m con −3 dB en B, con un pre-retardo d que conserva la oscilación previa al pulso (d = 3 para K = 16; d = 6 para K = 32):

$$
h[k] = \Big[\mathrm{IFFT}_L\big\{H_P(f_\ell)\,e^{-j2\pi f_\ell d T_s}\big\}\Big]_k,\qquad k = 0,\dots,K-1
$$

```python
def perfil_analogico(tipo, B, fs, m, K, d=None, L=4096):
    """Bessel o Butterworth de orden m con -3 dB en B, desde el prototipo analógico."""
    if d is None:
        d = 6 if K >= 32 else 3
    wc = 2 * np.pi * B
    if tipo == "bessel":
        b, a = signal.bessel(m, wc, analog=True, norm="mag")   # norm='mag': B es el punto de -3 dB
    elif tipo == "butter":
        b, a = signal.butter(m, wc, analog=True)
    else:
        raise ValueError(tipo)
    f = np.fft.fftfreq(L, d=1 / fs)
    _, H = signal.freqs(b, a, worN=2 * np.pi * f)
    H = H * np.exp(-2j * np.pi * f * d / fs)
    return normalizar_energia(np.fft.ifft(H).real[:K])

def K_regla(B, Rs, m):
    """K = 32 si B <= 0,4 Rs y m >= 4; K = 16 en los demás casos."""
    return 32 if (B <= 0.4 * Rs and m >= 4) else 16
```

> ⚠️ **Errores a evitar:**
> - No usar `signal.bessel(m, Wn)` ni `signal.butter(m, Wn)` digitales (transformación bilineal): a 2 sps el corte queda cerca de Nyquist y el Bessel pierde la fase lineal (error en banda de 0,53 dB y 0,78 muestras, contra 0,04 dB y 0,03 muestras en frecuencia).
> - No muestrear la respuesta al impulso continua del gaussiano: con B ≥ 0,5 R<sub>s</sub> produce *aliasing* y el −3 dB se corre (−0,33 dB con B = 0,8 R<sub>s</sub>).
> - Con `scipy.signal.bessel`, usar siempre `norm='mag'`; con la normalización por defecto, B no es el punto de −3 dB.

**Perfil CD corta** (Savory, 2008), usa `cd_R` de la Parte B:

```python
def perfil_cd_corta(fs, D, lam, z):
    """Taps de Savory para pocos km."""
    Ts = 1 / fs
    R = cd_R(fs, D, lam, z)
    n = np.arange(-(R // 2), R // 2 + 1)
    a = D * lam**2 * z
    return np.sqrt(1j * C * Ts**2 / a) * np.exp(-1j * np.pi * C * Ts**2 * n**2 / a)
```

### A.4 Parámetros configurables

| Parámetro | Rango | Por defecto |
|---|---|---|
| Perfil | bypass, eco, gaussiano, bessel, butter, medido, cd_corta | bypass |
| K | 1 a 32 | 1 (bypass); 16 o 32 con perfiles pasa-bajos (regla de K) |
| B (−3 dB) | 0,3 a 1,0 R<sub>s</sub> | 0,6 R<sub>s</sub> |
| m (orden) | 2 a 6 | 4 |
| Amplitud de eco \|aᵢ\| | 0,03 a 0,5 | — |
| Retardo de eco dᵢ | 1 a K − 1 muestras | — |

B se expresa como fracción de R<sub>s</sub> y se convierte con el `Escenario`, así el perfil vale para cualquier velocidad. Con m = 2 a 2 sps queda un error residual de hasta ≈ 0,6 dB en la banda (límite del muestreo, no del método).

### A.5 Bloque

```python
def isi_aplicar(x, h):
    """FIR complejo causal: y[n] = sum_k h[k] x[n-k] (mismo largo que x)."""
    return np.convolve(x, h)[: len(x)]
```

La clase `ISI(Bloque)` recibe un `ParametrosISI` (perfil, K, B, m, ecos o h), calcula h[k] una vez al construirse, expone `retardo = k₀` y `coeficientes() = {"h": h}`. Para procesar por bloques, conservar las últimas K − 1 muestras entre bloques (`scipy.signal.lfilter(h, 1, x, zi=...)`); el resultado debe ser idéntico al del procesamiento completo.

### A.6 Tests (`tests/test_isi.py`)

| Test | Criterio |
|---|---|
| Bypass: h = [1] devuelve la entrada | exacto |
| Impulso: δ[n] devuelve h[n] | exacto |
| Coincide con `numpy.convolve` (coeficientes complejos aleatorios) | ~10⁻¹² |
| Procesamiento por bloques = procesamiento completo | ~10⁻¹² |
| Linealidad | ~10⁻¹² |
| Energía normalizada: Σ\|h\|² = 1 | ~10⁻¹² |
| Perfiles pasa-bajos: \|H\| en f = B | −3,0 ± 0,3 dB |
| Truncamiento: energía capturada con K coeficientes | ≥ 98,5 % |
| Bessel/Butterworth frente al prototipo analógico (donde \|H\| > −20 dB) | < 0,5 dB y < 0,4 muestras (m ≥ 3) |
| Gaussiano: coeficientes simétricos respecto de k₀ | ~10⁻¹² |
| Los perfiles escalan con el escenario (mismo B/R<sub>s</sub> a 32 y 100 GBd da el mismo h) | ~10⁻¹² |

### A.7 Gráficos (notebook 02)

- Magnitud, retardo de grupo y coeficientes de gaussiano, Bessel y Butterworth con el mismo B.
- Efecto del orden m (magnitud, retardo de grupo, respuesta al escalón).
- Diseño en frecuencia frente a bilineal (para documentar la decisión).
- Constelación QAM16 y ojo PAM4 para B = 0,4; 0,5 y 0,6 R<sub>s</sub>.
- SNR equivalente de la ISI por perfil, en el instante óptimo y sobre la grilla fija.

---

## Parte B · Dispersión cromática

### B.1 Modelo físico

Desarrollando la constante de propagación alrededor de la portadora y descartando β₀ (fase constante) y β₁ (retardo común):

$$
H_{\mathrm{CD}}(\omega) = \exp\!\left[-j\left(\frac{\beta_2}{2}\,\omega^2 + \frac{\beta_3}{6}\,\omega^3\right) z\right],
\qquad
\beta_2 = -\frac{D\lambda^2}{2\pi c},
\qquad
\beta_3 = \left(\frac{\lambda^2}{2\pi c}\right)^{2}\left(S + \frac{2D}{\lambda}\right)
$$

Con β₃ = 0, y en la **convención positiva** adoptada:

$$
H_{\mathrm{CD}}(f) = \exp\!\left(+j\,\frac{\pi D\lambda^2 z}{c}\,f^2\right)
$$

que coincide con la ecuación (1) de Jiang et al. (2026). **No lleva el factor ½** que aparece en la guía del proyecto.

Propiedades (todas se testean): |H| = 1 (pasa-todo); retardo de grupo τ(f) = −(Dλ²z/c)·f cuando β₃ = 0; H<sup>−1</sup> = H<sup>*</sup>; H(z₁)·H(z₂) = H(z₁ + z₂).

Referencias: Mello y Barbosa, sec. 3.2; Agrawal, *Nonlinear Fiber Optics*, cap. 3; Jiang et al. (2026).

```python
def betas(D, lam, S=0.0):
    """beta2 [s^2/m] y beta3 [s^3/m] a partir de D [s/m^2], lam [m], S [s/m^3]."""
    k = lam**2 / (2 * np.pi * C)
    beta2 = -D * k
    beta3 = k**2 * (S + 2 * D / lam) if S != 0.0 else 0.0
    return beta2, beta3

def cd_transfer(L, fs, D, lam, z, S=0.0):
    """H_CD en la grilla de una FFT de L puntos (orden de numpy.fft). Convención positiva."""
    w = 2 * np.pi * np.fft.fftfreq(L, d=1 / fs)
    b2, b3 = betas(D, lam, S)
    return np.exp(-1j * (b2 / 2 * w**2 + b3 / 6 * w**3) * z)
```

### B.2 Longitud del chirp y tamaño de bloque

La respuesta al impulso es un chirp infinito y no causal. La cantidad de muestras que ocupa dentro de la banda de Nyquist (Savory, 2008; Jiang et al., ec. 3) es

$$
R = 2\left\lfloor \frac{|D|\lambda^2 z}{2cT_s^2}\right\rfloor + 1
\qquad\Longrightarrow\qquad N \ge R
$$

```python
def cd_R(fs, D, lam, z):
    """Longitud del chirp en muestras."""
    Ts = 1 / fs
    return 2 * int(np.floor(abs(D) * lam**2 * z / (2 * C * Ts**2))) + 1

def cd_N(fs, D, lam, z, Nmin=128, Nmax=2048):
    """Menor potencia de 2 >= R, acotada al rango configurable."""
    R = cd_R(fs, D, lam, z)
    N = max(Nmin, 1 << int(np.ceil(np.log2(R))))
    if N > Nmax:
        raise ValueError(f"R = {R} excede N_max = {Nmax}")
    return N
```

R crece linealmente con z y **cuadráticamente con f<sub>s</sub>**: en escenarios de mayor velocidad, `Nmax` tiene que acompañar (a 200 GSa/s y 1000 km, R ≈ 5 450).

| z [km] | R a 40 GSa/s | R a 64 GSa/s | N mínimo a 64 GSa/s |
|---|---|---|---|
| 10 | 3 | 5 | 128 |
| 100 | 21 | 55 | 128 |
| 500 | 109 | 279 | 512 |
| 1000 | 217 | 559 | 1024 |
| 2000 | 435 | 1117 | 2048 |

### B.3 Referencia ideal

Una única FFT del largo de toda la señal, con al menos R ceros a cada lado para que la convolución circular no contamine el resultado:

```python
def cd_referencia(x, fs, D, lam, z, S=0.0):
    """Referencia: una FFT del largo de toda la señal, con ceros a ambos lados."""
    pad = cd_R(fs, D, lam, z) + 64
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    L = 1 << int(np.ceil(np.log2(len(xp))))
    y = np.fft.ifft(np.fft.fft(xp, L) * cd_transfer(L, fs, D, lam, z, S))[: len(xp)]
    return y[pad : pad + len(x)]
```

> ⚠️ No hacer `ifft(fft(x) * H)` sobre la señal sin relleno: las colas del chirp "dan la vuelta" y contaminan el comienzo.

### B.4 Modelo por bloques: overlap-save

Es el algoritmo del hardware. Se evalúa H<sub>CD</sub> en una FFT de L = 2N puntos y se le agrega una fase lineal que centra el chirp en la ventana (Jiang et al., ec. 7, en la convención de `numpy`):

$$
H[k] = H_{\mathrm{CD}}(2\pi f_k), \qquad \bar{H}[k] = H[k]\,e^{-j2\pi k (N/2)/L}, \qquad L = 2N
$$

Por bloque: N muestras previas + N nuevas → FFT → × H̄ → IFFT → descartar las primeras N. La salida queda **retrasada N/2 muestras** (retardo puro).

```python
def cd_hbar(N, fs, D, lam, z, S=0.0):
    """Filtro de 2N puntos para overlap-save, con retardo de N/2."""
    L = 2 * N
    k = np.fft.fftfreq(L) * L
    return cd_transfer(L, fs, D, lam, z, S) * np.exp(-2j * np.pi * k * (N / 2) / L)

def cd_overlap_save(x, Hbar):
    """Overlap-save con FFT de L = 2N. La salida queda retrasada N/2 muestras."""
    L = len(Hbar)
    N = L // 2
    nb = int(np.ceil(len(x) / N))
    xz = np.concatenate([np.zeros(N, complex), x, np.zeros(nb * N - len(x), complex)])
    y = np.empty(nb * N, complex)
    for b in range(nb):
        y[b * N : (b + 1) * N] = np.fft.ifft(np.fft.fft(xz[b * N : b * N + L]) * Hbar)[N:]
    return y
```

`cd_hbar` es lo que calculará el host; `cd_overlap_save` es lo que hará la FPGA. Mantenerlos separados facilita la cuantización y la exportación. La clase `CD(Bloque)` recibe un `ParametrosCD` (D, λ, z, S, N opcional), elige N con `cd_N` si no se indica, y expone `retardo = N/2` y `coeficientes() = {"Hbar": Hbar}`.

Alcance esperado a 64 GSa/s (NRMSE < 10⁻² frente a la referencia): N = 128 → ≈ 200 km; 256 → 300 a 400 km; 512 → 800 a 900 km; 1024 → ≈ 1500 km; 2048 → 2000 km.

### B.5 Parámetros configurables

| Parámetro | Rango | Por defecto | Comentario |
|---|---|---|---|
| D | 2 a 23 ps/nm/km (se admite negativo) | 17 ps/nm/km | Negativo: fibra compensadora y test de reversibilidad. |
| λ | 1530 a 1565 nm | la del `Escenario` | |
| z | 0 a 2000 km | 100 km | |
| S | 0 a 0,092 ps/nm²/km | 0 | Activa β₃; típico SSMF 0,057. |
| N | 128 a `Nmax` | automático (`cd_N`) | `Nmax` depende del escenario. |

### B.6 Tests (`tests/test_cd.py`)

| Test | Criterio |
|---|---|
| z = 0 es la identidad | ~10⁻¹⁵ |
| Pasa-todo: \|H\| = 1 y energía conservada | ~10⁻¹² |
| **Convención de signo positiva**: fase de H = +πDλ²z f²/c a baja frecuencia | ~10⁻¹² |
| Pendiente del retardo de grupo = −Dλ²z/c (detecta el factor ½ y el signo) | error relativo < 0,1 % |
| Reversibilidad: (D, S) seguido de (−D, −S) | NRMSE ~10⁻¹² |
| Composición: H(z₁)·H(z₂) = H(z₁ + z₂) | ~10⁻¹² |
| Retardo de grupo con β₃: parábola con coeficientes β₂z y β₃z/2 | error relativo < 1 % |
| Compatibilidad: con S = 0, salida idéntica al modelo sin β₃ | exacto |
| Pulso gaussiano: ancho medido = T₀·√(1 + (z/L<sub>D</sub>)²), L<sub>D</sub> = T₀²/\|β₂\| | error relativo < 1 % |
| R = 3 y 109 a 40 GSa/s para 10 y 500 km | exacto (valores del paper) |
| Overlap-save frente a la referencia, con N ≥ R | NRMSE < 10⁻² |
| Retardo declarado del bloque = N/2 | exacto |

Ejemplo:

```python
# tests/test_cd.py
import numpy as np
from emulador.config import C, D_si, Escenario
from emulador.cd.fisica import cd_transfer, cd_R, cd_N
from emulador.cd.referencia import cd_referencia
from emulador.cd.overlap_save import cd_hbar, cd_overlap_save
from emulador.metricas import nrmse, retardo_de_grupo

ESC = Escenario()
D, LAM, FS = D_si(17.0), ESC.lam, ESC.fs

def test_convencion_signo_positiva():
    z, L = 10e3, 4096
    H = cd_transfer(L, FS, D, LAM, z)
    f = np.fft.fftfreq(L, d=1 / FS)
    k = np.abs(f) < 2e9                      # fase < pi: sin ambigüedad
    assert np.allclose(np.angle(H[k]), np.pi * D * LAM**2 * z * f[k] ** 2 / C)

def test_pendiente_retardo_de_grupo():
    z = 100e3
    f, tau = retardo_de_grupo(cd_transfer(8192, FS, D, LAM, z), FS)
    assert np.isclose(np.polyfit(f, tau, 1)[0], -D * LAM**2 * z / C, rtol=1e-3)

def test_overlap_save_contra_referencia():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)) / np.sqrt(2)
    z = 500e3
    N = cd_N(FS, D, LAM, z)
    y = cd_overlap_save(x, cd_hbar(N, FS, D, LAM, z))
    r = cd_referencia(x, FS, D, LAM, z)
    m = len(x) - 2 * N
    assert nrmse(y[N // 2 : N // 2 + m], r[:m]) < 1e-2
```

### B.7 Gráficos (notebooks 03 y 04)

- Magnitud (= 1), fase parabólica y retardo de grupo lineal para 10, 50 y 100 km.
- Pulso gaussiano: intensidad para varias z y ancho medido contra la curva teórica.
- Constelación QAM16 con CD creciente, y con CD seguida de H\* (reversibilidad).
- Respuesta al impulso (chirp) y tabla de R.
- NRMSE contra distancia para N = 128 a 2048 (y el FIR truncado como comparación).
- Residuo de β₃ tras compensar solo β₂: constelaciones y SNR equivalente contra R<sub>s</sub>.

### B.8 Implicancias del hardware (RFSoC)

La implementación será en un **RFSoC Zynq UltraScale+**; el modelo exacto no está confirmado. Según la hoja de datos de AMD (DS889), la familia va de 1 200 a 4 272 bloques DSP según el dispositivo, contra 240 de la Arty A7-100T.

- **Qué cambia:** la FFT radix-2² puede implementarse con arquitectura paralela segmentada (MDC) de P muestras por ciclo con P > 1. La tasa efectiva del bloque es R<sub>CD</sub> = P·f<sub>clk</sub>·N/L = P·f<sub>clk</sub>/2.
- **Qué no cambia:** la versión totalmente paralela del paper de Jiang ocupa unos 4 370 DSP solo para la CD, más que el RFSoC más grande. P sigue siendo el parámetro de compromiso entre velocidad y recursos, y se fija cuando se confirme el dispositivo.
- **Para el golden model:** el modelo en flotante no depende del hardware. El modelo en punto fijo debe replicar las etapas radix-2² y el escalado por etapa de la FFT elegida. Con el mismo escalado, P cambia cuántas muestras se procesan por ciclo pero no la aritmética de cada muestra (a confirmar con el diseño RTL).
- Los parámetros del hardware (P, f<sub>clk</sub>, N<sub>max</sub>) se guardan en una estructura separada del `Escenario`, porque describen la implementación y no el canal.

---

## Parte C · Etapas comunes

### C.1 Estímulos (`emulador/estimulos/`)

**Señales de prueba física:** impulso δ[n] (valida h[n]); pulso gaussiano de ancho T₀ (valida el ensanchamiento de la CD); tonos (validan \|H\| y la fase).

**Señales de comunicación:**
1. Bits de una **PRBS** (PRBS-15 o PRBS-31, LFSR), la misma que generará el hardware.
2. **Mapper** PAM4 o QAM16 con codificación Gray, potencia media unitaria.
3. **Sobremuestreo** a `Escenario.sps` y **conformación RRC** (roll-off del escenario, span ≥ 32 símbolos).
4. Se devuelven la señal y los **símbolos transmitidos**.

Cada estímulo se guarda con un `meta.json`: escenario, semilla, formato, largo y versión del paquete.

### C.2 Receptor mínimo y métricas

El receptor mínimo **no es parte del emulador**: aplica el filtro adaptado, submuestrea en el instante óptimo y normaliza, solo para visualizar y medir.

```python
def nrmse(y, ref):
    return np.sqrt(np.sum(np.abs(y - ref) ** 2) / np.sum(np.abs(ref) ** 2))

def retardo_de_grupo(H, fs):
    """tau(f) a partir de H en orden de numpy.fft. Devuelve (f, tau) ordenados."""
    f = np.fft.fftshift(np.fft.fftfreq(len(H), d=1 / fs))
    fase = np.unwrap(np.angle(np.fft.fftshift(H)))
    return f, -np.gradient(fase, 2 * np.pi * f)
```

Además: EVM y SNR equivalente (−20·log₁₀ EVM, con ganancia y fase óptimas), ancho rms de un pulso, distorsión de pico y potencia de ISI relativa al cursor.

### C.3 Cuantización (`emulador/cuantizacion.py`)

Se hace cuando el modelo en flotante está validado:

1. Formatos Qm.n para la señal de entrada, los coeficientes (h[k] o H̄[k]) y los intermedios.
2. `cuantizar(x, m, n, redondeo="nearest", saturar=True)`, aplicada por separado a la parte real y la imaginaria.
3. Barrido de bits y curvas NRMSE contra bits, separando referencia → flotante → punto fijo.
4. **ISI:** cuantizar h[k] y la entrada; el acumulador del FIR necesita log₂K bits de crecimiento.
5. **CD:** replicar la FFT radix-2² del hardware (etapas y escalado). Punto de partida de Jiang et al.: 10 bits para la señal (Q2.8) y 10 bits para H̄[k], con NRMSE ≈ 2·10⁻³ hasta 400 km.

### C.4 Exportación de vectores (`emulador/exportar.py`)

Para cada caso, en `vectores/<isi|cd>/<caso>/`:

| Archivo | Contenido |
|---|---|
| `entrada.npy` / `entrada.hex` | Señal de entrada (I y Q); el `.hex` es legible con `$readmemh`. |
| `coeficientes.npy` / `.hex` | h[k] (ISI) o H̄[k] (CD). |
| `salida_esperada.npy` / `.hex` | Salida del golden model. |
| `meta.json` | Escenario, parámetros del bloque, semilla, formato Qm.n, retardo y versión. |

Mientras no esté definido el punto fijo, solo `.npy` y `meta.json`.

---

## 7. Notebooks

Los notebooks son para **explicar y mostrar**: teoría con ecuaciones, gráficos y conclusiones en un mismo documento. Todo cálculo que se repita se mueve a `emulador/` y se importa.

| Notebook | Contenido |
|---|---|
| `00_escenario_y_convenciones.ipynb` | Escenario, unidades, convención de signo, magnitudes derivadas (β₂, β₃, R, L<sub>D</sub>). |
| `01_estimulos.ipynb` | PRBS, mapper y RRC: constelaciones, espectros y ojos limpios. |
| `02_isi_generica.ipynb` | Parte A. |
| `03_cd_modelo_fisico.ipynb` | Parte B, secciones B.1 a B.3. |
| `04_cd_overlap_save.ipynb` | Parte B, secciones B.4 y B.8. |
| `05_cuantizacion.ipynb` | Parte C.3. |
| `06_vectores_rtl.ipynb` | Parte C.4. |

**Estructura de cada notebook:** 1) Objetivo; 2) Teoría (LaTeX y referencia); 3) Parámetros (una sola celda, con el `Escenario`); 4) Simulación; 5) Validación (`assert` con los criterios de los tests); 6) Gráficos (guardados en `docs/figuras/`); 7) Conclusiones.

Primera celda de código:

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("../emulador/estilo.mplstyle")

from emulador.config import C, D_si, S_si, Escenario
esc = Escenario()   # cambiar acá para otro escenario
```

**Buenas prácticas:**
- `nbstripout` y `jupytext` (formato *percent*) para versionar los notebooks con diffs legibles.
- "Restart & Run All" antes de commitear o exportar.
- `jupyter nbconvert --to html notebooks/02_isi_generica.ipynb` para informes de avance; las figuras de `docs/figuras/` van directo al template de presentaciones (misma paleta: `#14213D`, `#B35418`, `#2B5C9E`).

---

## 8. Checklist

**Base**
- [ ] `pyproject.toml`, paquete instalable, `pytest` corriendo.
- [ ] `config.py` con `Escenario` y conversiones; `base.py`; `metricas.py`.
- [ ] Estímulos y receptor mínimo (notebook 01).

**ISI genérica**
- [ ] Perfiles y bloque `ISI`.
- [ ] Tests de la tabla A.6.
- [ ] Notebook 02 con los gráficos de A.7.

**Dispersión cromática**
- [ ] `fisica.py` y `referencia.py` con los tests de física de B.6 (incluido el de signo).
- [ ] `overlap_save.py` y bloque `CD`; test contra la referencia.
- [ ] Notebooks 03 y 04 con los gráficos de B.7.

**Hardware y RTL**
- [ ] Confirmar el modelo de RFSoC y fijar P, f<sub>clk</sub> y N<sub>max</sub>.
- [ ] Cuantización (notebook 05).
- [ ] Exportación de vectores (notebook 06).
- [ ] Comunicar la convención de signo positiva a la tesis de CDC.

---

## 9. Referencias

- D. A. A. Mello y F. A. Barbosa, *Digital Coherent Optical Systems*, Springer, 2021. Cap. 3.
- G. P. Agrawal, *Fiber-Optic Communication Systems*, Wiley. Cap. 2.
- G. P. Agrawal, *Nonlinear Fiber Optics*, Academic Press. Cap. 3.
- J. G. Proakis y M. Salehi, *Digital Communications*, McGraw-Hill.
- J. G. Proakis y D. G. Manolakis, *Digital Signal Processing*, Pearson.
- S. J. Savory, "Digital filters for coherent optical receivers", *Optics Express*, 16(2), 2008.
- J. Jiang, C. Xie y M. Chen, "Hardware implementation of configurable chromatic dispersion model for real-time optical fiber communication simulation systems", *J. Lightwave Technol.*, 44(12), 2026.
- E. Börjeson y P. Larsson-Edefors, "Fiber-on-Chip: Digital emulation of channel impairments for real-time DSP evaluation", *J. Lightwave Technol.*, 41(3), 2023.
- S. He y M. Torkelson, "A new approach to pipeline FFT processor", *Proc. IPPS*, 1996.
- AMD, *Zynq UltraScale+ RFSoC Data Sheet: Overview* (DS889).
- Informe interno: *Fundamentos teóricos para el modelado de dispersión cromática e ISI* (PROCOM, 2026).