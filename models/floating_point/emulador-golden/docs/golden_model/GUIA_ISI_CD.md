# Guía de trabajo · Golden model de ISI genérica y dispersión cromática

Guía para completar, **dentro de este repositorio**, el golden model de los bloques ISI genérica (01) y dispersión cromática (03). Rama de trabajo: `wanda_dev`.

- Referencia técnica: [referencia_guia_ISI_CD.md](referencia_guia_ISI_CD.md) (en esta guía, "Ref. §X").
- Fundamento teórico: [Fundamentos_CD_ISI_golden_model.pdf](Fundamentos_CD_ISI_golden_model.pdf) (en esta guía, "PDF Ec. N").
- Decisiones: §1 de esta guía. Avance: checklist de §4 y `git log`.

Todas las rutas son relativas a `models/floating_point/emulador-golden/`, salvo que se indique otra cosa.

---

## 1. Decisiones vigentes

Esta tabla es el **único registro de decisiones**:
- una decisión nueva se agrega como fila, en el mismo commit que la tarea que la necesitó;
- si una decisión cambia, se edita su fila y el motivo va en el mensaje del commit;
- una fila "sin confirmar" no se aplica al código hasta que Wanda la confirme.

| # | Tema | Decisión | Estado |
|---|---|---|---|
| 1 | Escenario de referencia | 32 GBd, 2 sps, f<sub>s</sub> = 64 GSa/s, **configurable** mediante `Scenario` (`emulador/config.py`) | aceptada |
| 2 | Signo de la CD | **Positivo**: H<sub>CD</sub>(f) = exp(+jπDλ²z f²/c), equivalente a exp(−jβ(ω)z) con β₂ = −Dλ²/(2πc). Hay que comunicarlo a la tesis de CDC, que debe usar H* | aceptada |
| 3 | Factor ½ de la guía original del proyecto | **Descartado**: emularía la mitad de la dispersión | aceptada |
| 4 | Término β₃ | Incluido, con S = 0 por defecto (con S = 0, β₃ = 0 exacto) | aceptada |
| 5 | Método de la CD | Muestreo en frecuencia + overlap-save; FFT de L = 2N; retardo N/2; N ≥ R; N automático = max(128, menor potencia de 2 ≥ R), hasta N<sub>max</sub> | aceptada |
| 6 | ISI | FIR complejo causal, K ≤ 32, energía normalizada. Coeficientes diseñados en frecuencia: Bessel y Butterworth desde el prototipo analógico, **nunca con la transformación bilineal**. Perfil principal: Bessel m = 4, B = 0,6 R<sub>s</sub>, K = 16. Regla: K = 32 si B ≤ 0,4 R<sub>s</sub> y m ≥ 4. Por defecto, el bloque queda en bypass | aceptada |
| 7 | Hardware | RFSoC Zynq UltraScale+ (modelo sin confirmar); FFT radix-2² con paralelismo P parametrizable. P, f<sub>clk</sub> y N<sub>max</sub> son **provisorios** (`HardwareParams`). Reemplaza la suposición de Artix-7 del PDF | aceptada (parcial) |
| 8 | Retardo | En ISI y CD, `process()` devuelve la salida **alineada** (descuenta el retardo entero; el fraccionario no se corrige). El bloque expone `delay_samples`, `coefficients()` y `process_hw()` (salida cruda, como el hardware, que es la que se exporta) | aceptada |
| 9 | Extensión de `Block` | Agregar esas tres piezas a `common.Block` con valores neutros. Hasta que haya acuerdo, se definen solo en ISI y CD | confirmada; falta acuerdo del equipo |
| 10 | Estímulos | Módulo nuevo `emulador/stimuli/`, sin tocar `common.py` | confirmada; falta acuerdo del equipo |
| 11 | Notebooks | `notebooks/` + extra `notebooks` en `pyproject.toml` (`jupyterlab`, `ipykernel`, `jupytext`, `nbstripout`, `nbconvert`). Cada máquina corre `nbstripout --install` una vez (config local de git) | aplicada en `wanda_dev`; falta PR al equipo |
| 12 | Entorno | venv por máquina en `.venv/` (ignorado); Docker más adelante | aceptada |
| 13 | Estructura | Prevalecen las convenciones del repo sobre la referencia (§2) | aceptada |
| 14 | Punto fijo | Modelo bit a bit en `models/fixed_point/emulador-real/`; en el golden solo utilidades y exploración | **sin confirmar** |
| 15 | RRC de los estímulos y del receptor | span = 64 símbolos por defecto (`shape`, `comm_signal`, `matched_filter`). Piso de EVM back-to-back medido en P3.3 con roll-off 0,1 y 2 sps: span 32 → 0,58 %, 48 → 0,27 %, **64 → 0,12 %**, 96 → 0,04 %. Criterio back-to-back: EVM < 0,5 % | aceptada |
| 16 | Convención de la PRBS | LFSR de Fibonacci según ITU-T O.150 (x<sup>m</sup> + x<sup>k</sup> + 1): b[n] = b[n−m] ⊕ b[n−k]. Los primeros m bits de salida son la semilla, del MSB al LSB. Sin inversión por defecto; `invert=True` da la salida invertida que O.150 pide para PRBS-15/23/31. El generador del RTL tiene que usar la misma convención | aceptada |
| 17 | Orden de trabajo de ISI y CD | **Intercalado por paso**, para presentar avances de los dos efectos: A1 → B1 → A2 → B2 → A3 → A4 → B3 → A5 → B4, agrupados en 4 hitos (checklist de §4). Se respetan las dependencias de cada paso | aceptada |
| 18 | Referencia ideal de la CD (`cd_reference`) | Con `full=True` devuelve **las L muestras** del buffer de la FFT (x[n] ↔ y[pad + n], pad = R + 64 = `cd_reference_pad`), así la energía se conserva (rel. ~1e-16). Reversibilidad (D, S) → (−D, −S): **NRMSE < 1e-5**, no 1e-12: las colas del chirp discreto decaen como ~1/n, no entran en el relleno y dan la vuelta. El piso crece con z y baja con el largo de la señal (4096 símbolos QAM16: 1,0e-6 a 100 km, 2,8e-6 a 500 km, 7,3e-6 a 2000 km; con 2048 símbolos, 2,0e-5 a 2000 km). Sobre la misma grilla de L puntos, H* deshace H exacto (1e-12) | aceptada |
| 19 | Criterios de Bessel y Butterworth (A3) | A 2 sps, con B > 0,5·R<sub>s</sub>, el prototipo analógico no está limitado en banda en f<sub>s</sub>/2 y el −3 dB se corre (hasta 0,66 dB con m = 2; 0,4 dB con Bessel m = 6, B = 0,6·R<sub>s</sub>); a 4 sps, con K y d al doble, el desvío baja a 0,11 dB: es el muestreo, no el diseño. Criterios: **−3 dB ±0,3 dB y energía ≥ 98,5 %** a 2 sps para B ≤ 0,5·R<sub>s</sub> (m = 2…6) y para el perfil principal (B = 0,6·R<sub>s</sub>, m = 4), y a 4 sps para B = 0,3…0,8·R<sub>s</sub>. **Fidelidad** (< 0,5 dB y < 0,4 muestras) en la banda pasante \|f\| ≤ B, no en \|H<sub>P</sub>\| > −20 dB: cerca de Nyquist el FIR no puede seguir al prototipo y hay ceros profundos. Vale para m ≥ 3 y B ≤ 0,7·R<sub>s</sub> (Bessel m = 3, B = 0,8·R<sub>s</sub>: 0,82 dB). Perfil principal: 0,06 dB y 0,03 muestras | aceptada |

---

## 2. Mapeo de la referencia al repositorio

La Ref. §4 propone una estructura genérica. Acá se adapta al repositorio, que tiene prioridad.

### 2.1 Convenciones del repo que prevalecen sobre la referencia

| Referencia | Repositorio (se adopta) | Motivo |
|---|---|---|
| Identificadores en castellano (`Bloque`, `retardo`, `Escenario`, `cd_referencia`) | **Identificadores en inglés** (`Block`, `delay_samples`, `Scenario`, `cd_reference`); docstrings y comentarios en castellano | Convención de `emulador/` |
| Clase `Bloque(fs)` en `base.py` con `__call__` | `common.Block` (dataclass): se implementa `process()`, y `__call__` maneja bypass y forma | Interfaz común del equipo |
| `bloque.retardo` | `delay_samples` (propiedad del bloque) | Nombre en inglés; ver decisión 8 de §1 |
| `bloque.coeficientes()` | `coefficients()` | Idem |
| Parámetros del bloque en SI | **Campos del bloque en unidades de hoja de datos con sufijo** (`D_ps_nm_km`, `wavelength_nm`, `length_km`, `S_ps_nm2_km`). Todo el cálculo interno (`emulador/cd/`, `emulador/isi/`) es en SI, y la conversión se hace en el bloque con `config.D_si`/`S_si` | Convención de los esqueletos; la regla "SI adentro" de la Ref. §3 se mantiene |
| Por defecto z = 100 km, perfil Bessel | **Por defecto, efecto nulo** (`length_km=0`, ISI en bypass). 100 km y Bessel m=4 son los casos de ensayo de tests y notebooks | `test_contract.py` exige efecto nulo por defecto |
| `ParametrosISI`, `ParametrosCD` | Campos de dataclass del bloque, más `from_scenario(...)` | Así se construyen los bloques del repo (`REGISTRY`, `run_block`) |
| Señal `(n,)` | `(n,)` o `(2, n)`; operar sobre el último eje | Contrato provisorio |
| Paquete `emulador`, extra `dev` con Jupyter | Paquete `emulador-golden` (sin cambios); Jupyter en un extra `notebooks` aparte | `pyproject.toml` es compartido |
| `vectores/` en la raíz del paquete | `outputs/vectores/` | `outputs/` ya está en `.gitignore`, así no se toca el ignore compartido |
| `docs/figuras/`, `docs/referencias/` | `docs/golden_model/figuras/`; el PDF y la referencia ya están en `docs/golden_model/` | Documentación propia en una sola carpeta |
| Nivel "punto fijo" dentro del golden | Solo utilidades y exploración en el golden; el modelo bit a bit va en `models/fixed_point/emulador-real/` | Estructura del repo (**a decidir**, ver §6) |
| README = guía | Esta guía; el `README.md` del paquete es compartido | — |

### 2.2 Módulo por módulo

| Referencia | Archivo real | Estado | Integración |
|---|---|---|---|
| `emulador/config.py` | `emulador/config.py` | crear | `C`, `D_si`, `S_si`, `Scenario(Rs, sps, wavelength, rolloff)` con `fs`/`Ts`, y `HardwareParams(P, f_clk, N_max)` (provisorio, separado del escenario) |
| `emulador/base.py` | `emulador/common.py` (`Block`) | existe, **no se modifica** | Mientras no se acepte el cambio en `Block`, `delay_samples`/`coefficients()`/`process_hw()` se definen en cada bloque |
| `emulador/metricas.py` | `emulador/metrics.py` | crear | Reusa `common.evm_percent`; agrega `nrmse`, `group_delay`, `snr_eq_db`, `rms_width`, `peak_distortion`, `isi_power_ratio` |
| `emulador/estimulos/{prbs,mapper,pulso,pruebas}.py` | `emulador/stimuli/{prbs,mapper,pulse,test_signals}.py` | crear | `pulse.py` reusa `common.rrc_taps`. `gen_signal` queda para `test_contract`/`run_block` |
| `emulador/rx.py` | `emulador/rx.py` | crear | Generaliza `common.recover_symbols` con la búsqueda del instante óptimo y el descuento de retardo |
| `emulador/graficos.py`, `estilo.mplstyle` | `emulador/plots.py`, `emulador/style.mplstyle` | crear | Reusa la idea de `scripts/run_block.plot_before_after`; paleta `#14213D`, `#B35418`, `#2B5C9E` |
| `emulador/isi/perfiles.py` | `emulador/isi/profiles.py` | crear | Funciones puras de diseño (SI) |
| `emulador/isi/bloque.py` | `emulador/isi/fir.py` + `emulador/blocks/isi.py` | crear / completar esqueleto | `fir.py`: `isi_apply`, `isi_apply_stream`. El bloque `ISI` sigue en `REGISTRY["isi"]` |
| `emulador/cd/fisica.py` | `emulador/cd/physics.py` | crear | `betas`, `cd_transfer`, `cd_R`, `cd_N`, `savory_taps`, `cd_summary`. El perfil ISI "CD corta" usa `savory_taps` |
| `emulador/cd/referencia.py` | `emulador/cd/reference.py` | crear | `cd_reference(x, fs, D, lam, z, S, full=False)` |
| `emulador/cd/overlap_save.py` | `emulador/cd/overlap_save.py` | crear | `cd_hbar` (host), `cd_overlap_save` (FPGA) |
| `emulador/cd/bloque.py` | `emulador/blocks/chromatic_dispersion.py` | completar esqueleto | Sigue en `REGISTRY["chromatic_dispersion"]`; corregir el signo del docstring y `fs` por defecto a 64e9 |
| `emulador/cuantizacion.py` | `emulador/quantization.py` | crear | Solo utilidades Qm.n y barridos exploratorios |
| `emulador/exportar.py` | `emulador/export.py` + `scripts/export_vectors.py` | crear | Escribe en `outputs/vectores/<isi\|cd>/<caso>/` |
| `tests/test_config.py`, `test_isi.py`, `test_cd.py` | `tests/test_config.py`, `tests/test_metrics.py`, `tests/test_stimuli.py`, `tests/test_isi.py`, `tests/test_chromatic_dispersion.py`, `tests/test_quantization.py`, `tests/test_export.py`, `tests/data/` (regresiones) | crear | `tests/test_contract.py` tiene que seguir pasando |
| `notebooks/0X_*.ipynb` | `notebooks/0X_*.ipynb` (nombres de la Ref. §7) | crear | Importan de `emulador/`; figuras en `docs/golden_model/figuras/` |
| `scripts/` | `scripts/` (existe `run_block.py`) | ampliar | `run_block isi --param profile=bessel` debe funcionar |

### 2.3 Interfaz de los dos bloques (propuesta de firma)

```python
@dataclass
class ISI(Block):
    taps: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))  # se usa si profile="taps"
    profile: str = "taps"      # taps | echo | gaussian | bessel | butter | measured | cd_short
    Rs: float = 32e9
    sps: int = 2
    B_rel: float = 0.6         # B / Rs
    order: int = 4             # m
    K: int | None = None       # None -> regla de K
    echoes: tuple = ()         # ((d_i, a_i), ...)
    name = "isi"
    # __post_init__: calcula self.h; delay_samples = k0 = argmax|h|
    # process(x): FIR y recorte de k0 (salida alineada)   process_hw(x): FIR causal sin recortar

@dataclass
class ChromaticDispersion(Block):
    fs: float = 64e9
    D_ps_nm_km: float = 17.0
    wavelength_nm: float = 1550.0
    length_km: float = 0.0
    S_ps_nm2_km: float = 0.0
    N: int | None = None       # None -> cd_N (max(128, pot. de 2 >= R))
    name = "chromatic_dispersion"
    # delay_samples = N/2; coefficients() = {"Hbar": Hbar}
    # process(x): OLS alineado (descuenta N/2)   process_hw(x): OLS crudo
```

`from_scenario(esc, **kw)` completa `Rs`, `sps`, `fs` y `wavelength` a partir de un `Scenario`.

---

## 3. Pasos

### 3.0 Cómo se trabaja

Cada paso (P1, A2, …) se divide en **tareas atómicas**, listadas en la checklist de §4. Una tarea es la unidad de trabajo entre las dos computadoras:

> **Una tarea = un commit = un push.** Al terminar una tarea el repositorio queda con los tests en verde y la casilla marcada. Así la otra computadora puede seguir con solo hacer `git pull`.

Ciclo de cada tarea:

1. `git pull --rebase` y elegir la primera tarea sin marcar de §4 (respetando las dependencias del paso).
2. **Leer** la sección del paso en §3 y la parte de la referencia y del PDF que indica. Verificar que no contradiga §1; si hace falta una decisión nueva, preguntarle a Wanda y agregarla a §1.
3. **Escribir los tests** con los criterios numéricos del paso.
4. **Implementar** en `emulador/`. La lógica vive en funciones puras en SI (`emulador/isi/`, `emulador/cd/`); los bloques de `emulador/blocks/` solo convierten unidades y llaman a esas funciones.
5. **Correr `pytest` completo.** `test_contract.py` tiene que seguir pasando.
6. **Marcar la casilla** de la tarea en §4 y hacer commit: `[golden-isi-cd] <ID> <descripción>` (por ejemplo, `[golden-isi-cd] P2.1 nrmse y group_delay`).
7. **Push**, preguntando antes.

Si hay que cortar a mitad de una tarea, commit `[golden-isi-cd] <ID> WIP: <qué falta>` y push; la casilla queda sin marcar. Los tests pueden quedar rotos solo en un commit WIP.

Cada paso tiene los mismos apartados:

| Apartado | Qué dice |
|---|---|
| Objetivo | El resultado en una línea |
| Depende de | Pasos que tienen que estar listos antes |
| Para qué | Por qué existe el paso y qué problema evita |
| Archivos | Rutas reales que se crean (+) o modifican (~) |
| Qué hacer | Subpasos en orden, con firmas y ecuaciones |
| Tests | Cómo se arma cada test y su criterio |
| Notebook | Qué se muestra y qué gráficos se guardan |
| Errores a evitar | Trampas conocidas |
| Verificar | Comandos para comprobarlo |
| Listo cuando | Condición de cierre |

Esfuerzo: **S** (≤ medio día), **M** (1 a 2 días), **L** (≥ 3 días). Una vez terminada la base común, las partes A y B son independientes, salvo A4, que usa una función de B1. Se avanzan intercaladas, en el orden de §4 (decisión 17).

Comandos habituales, desde `models/floating_point/emulador-golden/` y con el venv activo (`. .venv/bin/activate`):

```bash
pytest                                   # todo
pytest tests/test_isi.py -k gaussian -v  # un subconjunto
python -m scripts.run_block isi --param profile=bessel
jupyter nbconvert --to notebook --execute --inplace notebooks/02_isi_generica.ipynb   # "Restart & Run All" por consola
```

---

### Base común

#### P0 · Entorno y línea de base — S

- **Objetivo:** `pytest` corriendo en ASUS y en LENOVO con el mismo resultado.
- **Depende de:** nada.
- **Para qué:** si las dos máquinas no dan el mismo resultado de partida, más adelante no se sabe si una diferencia viene del código o del entorno.
- **Archivos:** ninguno versionado. `.venv/` es local y está ignorado.
- **Qué hacer:**
  1. `cd models/floating_point/emulador-golden`.
  2. `python3 -m venv .venv && . .venv/bin/activate`.
  3. `pip install -U pip && pip install -e ".[dev,notebooks]"`. El modo editable hace que los notebooks y los tests vean los cambios sin reinstalar.
  3b. `nbstripout --install`: borra las salidas de los notebooks al commitear. Es configuración local de git, así que va una vez por máquina.
  4. `pytest` y `python -m scripts.run_block phase_offset --param delta_theta_rad=0.3`.
  5. Anotar en la tabla de máquinas de `CLAUDE.md` (raíz del repo) las versiones de Python, numpy y scipy (`pip list | grep -Ei "numpy|scipy|matplotlib|pytest"`).
- **Tests:** los de `tests/test_contract.py`. Resultado esperado: los de `phase_offset` pasan y el resto de los bloques queda en `skipped`.
- **Notebook:** —
- **Errores a evitar:** instalar con el `pip` del sistema (hay que activar el venv antes); commitear `.venv/`; distintas versiones mayores de numpy o scipy en cada máquina.
- **Verificar:** `pytest -q` da la misma cuenta de passed/skipped en las dos máquinas.
- **Listo cuando:** las dos filas de la tabla de máquinas tienen las versiones y el resultado de `pytest`.

#### P1 · Escenario, unidades y parámetros de hardware — S

- **Objetivo:** una sola fuente para f<sub>s</sub>, T<sub>s</sub>, λ y roll-off, y las conversiones de unidades.
- **Depende de:** P0.
- **Para qué:** todo lo que depende de la velocidad (B de los perfiles, R y N de la CD) se deriva del escenario. Cambiar de 32 a 100 GBd no debe requerir tocar otro módulo (Ref. §2).
- **Archivos:** + `emulador/config.py`, + `tests/test_config.py`.
- **Qué hacer:**
  1. **Constante:** `C = 299_792_458.0` (m/s, exacta).
  2. **Conversiones:**
     - `D_si(D_ps_nm_km)`: 1 ps/(nm·km) = 10⁻¹² / (10⁻⁹ · 10³) = 10⁻⁶ s/m², así que 17 → 1,7·10⁻⁵;
     - `S_si(S_ps_nm2_km)`: 1 ps/(nm²·km) = 10⁻¹² / (10⁻¹⁸ · 10³) = 10³ s/m³, así que 0,057 → 57.
  3. **`@dataclass(frozen=True) class Scenario`** con `Rs=32e9`, `sps=2`, `wavelength=1550e-9` y `rolloff=0.1`, y las propiedades `fs = Rs·sps` y `Ts = 1/fs`.
     - Validar en `__post_init__`: R<sub>s</sub> ∈ [1, 300] GBd; sps ∈ {1, 2, 4, 8, 16}; λ ∈ [1530, 1565] nm; roll-off ∈ [0,05, 0,5].
     - Agregar `to_dict()` para guardarlo en los `meta.json`.
  4. **`@dataclass(frozen=True) class HardwareParams`** con `P=1`, `f_clk=None` y `N_max=2048`, y `provisional: bool = True`. Queda separado del escenario porque describe la implementación, no el canal (Ref. B.8).
  5. Docstring de módulo con la tabla de convenciones de la Ref. §3: SI, `fftfreq`, signo positivo y retardos.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Conversión de D | `D_si(17.0)` contra `1.7e-5` | rel. 10⁻¹² |
  | Conversión de S | `S_si(0.057)` contra `57.0` | rel. 10⁻¹² |
  | Escenario por defecto | `fs == 64e9`, `Ts == 1/64e9` | exacto |
  | Validación | `Scenario(Rs=500e9)`, `sps=3`, `wavelength=1600e-9` | `ValueError` |
  | Inmutable | asignar `esc.Rs = 1` | `FrozenInstanceError` |
  | Hardware provisorio | `HardwareParams().provisional` | `True` |

- **Notebook:** `00_escenario_y_convenciones.ipynb` (primera parte): tabla del escenario, unidades y convención de signo explicada en palabras. Las magnitudes derivadas se agregan en B1.
- **Errores a evitar:** convertir unidades en más de un lugar; usar `3e8` en vez de `C` (cambia R en algunos casos límite de la Tabla 6).
- **Verificar:** `pytest tests/test_config.py -v`.
- **Listo cuando:** los tests pasan y ningún módulo nuevo escribe `64e9` a mano.

#### P2 · Métricas — S

- **Objetivo:** las métricas que comparan los tres niveles (referencia, flotante, punto fijo) y que caracterizan la ISI.
- **Depende de:** P1.
- **Para qué:** todos los tests y notebooks posteriores usan estas funciones. Si una métrica está mal, todos los criterios quedan mal, por eso se testean contra casos analíticos.
- **Archivos:** + `emulador/metrics.py`, + `tests/test_metrics.py`.
- **Qué hacer** (firmas):
  1. `nrmse(y, ref)` = √(Σ|y − ref|² / Σ|ref|²) (Ref. §1, PDF Ec. 32). Exige el mismo largo; alinear es responsabilidad del que llama.
  2. `group_delay(H, fs) -> (f, tau)`: H en orden `fftfreq`; `fftshift`, `unwrap` de la fase y τ = −dφ/dω con `np.gradient(fase, 2πf)` (Ref. C.2). Devuelve f ordenada.
  3. `evm_opt(rx, ref)`: EVM con ganancia **y fase** óptimas, g = Σ rx*·ref / Σ|rx|², EVM = ‖g·rx − ref‖/‖ref‖.
     - Es distinta de `common.evm_percent`, que **no** corrige fase. Documentar la diferencia; las dos se usan.
  4. `snr_eq_db(rx, ref) = −20·log10(evm_opt)`.
  5. `rms_width(a, t)`: σ = √(Σt²|a|²/Σ|a|² − (Σt|a|²/Σ|a|²)²).
  6. `peak_distortion(p)`: D<sub>p</sub> = Σ<sub>n≠n₀</sub>|p[n]| / |p[n₀]|, con n₀ = argmax|p| (PDF Ec. 13).
  7. `isi_power_ratio(p)`: Σ<sub>n≠n₀</sub>|p[n]|² / |p[n₀]|². Su inversa en dB es la "SNR equivalente" de la Tabla 4 del PDF.
  8. `symbol_rate_response(h, sps, rolloff, span, phase)`: respuesta total RRC ∗ h ∗ RRC muestreada cada sps muestras a partir de `phase`. Es la p[n] de las dos métricas anteriores.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | NRMSE nulo y conocido | `nrmse(x, x)`; `nrmse(1.01*x, x)` | 0 y 0,01 (10⁻¹²) |
  | Retardo de grupo de un retardo puro | H = e^(−j2πf·d·T<sub>s</sub>) con d = 5 | τ = 5·T<sub>s</sub> (rel. 10⁻⁹) |
  | EVM con fase óptima | `rx = 2·e^(j0,7)·ref` | EVM = 0 (10⁻¹²) |
  | Ancho rms | a = exp(−t²/2T₀²), \|a\|² = exp(−t²/T₀²) | σ = T₀/√2 (rel. 10⁻³) |
  | Distorsión de pico e ISI | p = [0, 1, 0] y p = [1, a] | 0; \|a\|² (10⁻¹²) |
  | Respuesta a símbolo sin ISI | `symbol_rate_response([1], …)` | ISI < −40 dB (piso del RRC truncado) |

- **Notebook:** — (se usan en todos).
- **Errores a evitar:** calcular el retardo de grupo sin `unwrap`; mezclar orden `fftfreq` y `fftshift`; usar `evm_percent` donde hace falta corregir fase (la CD rota la constelación).
- **Verificar:** `pytest tests/test_metrics.py -v`.
- **Listo cuando:** los tests pasan y cada función tiene docstring con su ecuación y unidades.

#### P3 · Estímulos — M

- **Objetivo:** señales reproducibles de prueba física (impulso, pulso gaussiano, tonos) y de comunicación (PRBS → mapper Gray → RRC).
- **Depende de:** P1. Usa la decisión 10 de §1 (estímulos en módulo nuevo).
- **Para qué:** cada test necesita un estímulo específico. El impulso valida h[n], el gaussiano el ensanchamiento de la CD, los tonos \|H\| y φ, y QAM/PAM el efecto sobre una señal real. La PRBS es la misma que generará el hardware, así los vectores exportados son reproducibles en RTL (Ref. C.1).
- **Archivos:** + `emulador/stimuli/{__init__,prbs,mapper,pulse,test_signals}.py`, + `tests/test_stimuli.py`.
- **Qué hacer:**
  1. **`prbs.py`:** `prbs(order, n_bits, seed=1) -> np.ndarray[uint8]` con LFSR de Fibonacci según ITU-T O.150.
     - Polinomios: PRBS-15 = x¹⁵ + x¹⁴ + 1; PRBS-31 = x³¹ + x²⁸ + 1.
     - La semilla no puede ser 0.
  2. **`mapper.py`:** `map_gray(bits, mod)`.
     - PAM4: 00→−3, 01→−1, 11→+1, 10→+3, normalizado por √5.
     - QAM16: 2 bits para I y 2 para Q, cada uno como PAM4 Gray, normalizado por √10.
     - Error si `len(bits)` no es múltiplo de los bits por símbolo.
  3. **`pulse.py`:** `shape(symbols, esc, span=64)` inserta ceros (× sps) y convoluciona con `common.rrc_taps(esc.rolloff, span, esc.sps)` en modo `"same"`, como `gen_signal`. Normaliza a potencia media 1.
  4. **`CommSignal`** (dataclass): `samples`, `symbols`, `bits`, `scenario`, `seed`, `mod`, más `meta()`. Y `comm_signal(esc, mod="qam16", n_symbols=4096, prbs_order=15, seed=1)`.
  5. **`test_signals.py`:**
     - `impulse(n, pos)`;
     - `gaussian_pulse(T0, fs, n)` con A = exp(−t²/2T₀²) centrado;
     - `tone(k, n)` = exp(j2πkn/N), sobre la grilla de la FFT para que caiga en un solo bin.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Período de la PRBS-15 | generar 2·(2¹⁵−1) bits | se repite con período 2¹⁵−1 (exacto) |
  | Balance | contar unos en un período | 2¹⁴ (exacto) |
  | Gray | etiquetas de símbolos vecinos | difieren en 1 bit (exacto) |
  | Potencia | constelación completa | media 1 (10⁻¹²) |
  | Back-to-back | `comm_signal` → filtro adaptado ideal con roll-off 0,1 (en P4, con `recover`) | EVM < 0,5 % (decisión 15) |
  | Tono | `fft(tone(k, n))` | un solo bin ≠ 0 (10⁻¹²) |
  | Reproducible | misma semilla dos veces | idéntico (exacto) |

- **Notebook:** `01_estimulos.ipynb`: bits → símbolos → espectro del RRC (ancho (1+roll-off)·R<sub>s</sub>), constelación limpia y ojo PAM4.
- **Errores a evitar:** usar `rng.integers` en lugar de la PRBS para los vectores de RTL; olvidar normalizar la potencia (la ISI supone potencia 1); span corto que sube el piso de EVM (con 32 símbolos y roll-off 0,1 ya da 0,58 %; decisión 15).
- **Verificar:** `pytest tests/test_stimuli.py -v`.
- **Listo cuando:** los tests pasan y `CommSignal.meta()` incluye escenario, semilla, orden de PRBS, modulación y largo.

#### P4 · Receptor mínimo y gráficos — M

- **Objetivo:** recuperar símbolos para medir y graficar. **No es parte del emulador.**
- **Depende de:** P2, P3.
- **Para qué:** Bessel y Butterworth introducen un retardo con parte fraccionaria, y la CD rota la constelación. Sin buscar el instante óptimo ni descontar el retardo, la EVM medida refleja el receptor y no el canal (PDF §3.5.5: más de 10 dB de diferencia).
- **Archivos:** + `emulador/rx.py`, + `emulador/plots.py`, + `emulador/style.mplstyle`, ~ `tests/test_stimuli.py`.
- **Qué hacer:**
  1. `matched_filter(x, esc, span=64)`: convolución con el mismo RRC.
  2. `recover(x, sig, search=True) -> RxResult(symbols, phase, delay, evm)`:
     - para cada fase de muestreo 0 … sps−1, submuestrear;
     - estimar el retardo entero en símbolos por correlación cruzada con `sig.symbols`;
     - quedarse con la combinación de menor `evm_opt`.
  3. Para análisis fino del instante óptimo, generar la señal con `Scenario(sps=16)`: los perfiles escalan con B/R<sub>s</sub>.
  4. `plots.py`:
     - `constellation(ax, s)`;
     - `eye(ax, x, sps, n_traces)`, con ventanas de 2 símbolos superpuestas;
     - `spectrum(ax, x, fs)` con `welch` de dos lados;
     - `save_fig(fig, name)` → `docs/golden_model/figuras/<name>.png` (y `.pdf` para presentaciones).
  5. `style.mplstyle` con la paleta `#14213D`, `#B35418`, `#2B5C9E`, grilla suave y tamaños de fuente para diapositivas.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Back-to-back | `comm_signal` → `recover` | EVM < 0,5 % |
  | Retardo entero | desplazar la señal 7 muestras | `delay` y `phase` recuperan el 7; EVM vuelve al piso |
  | Fase de portadora | multiplicar por e^(j0,5) | `evm_opt` en el piso |

- **Notebook:** `01_estimulos.ipynb`: ojo PAM4 y constelación QAM16 limpios con el estilo del proyecto.
- **Errores a evitar:** usar `common.recover_symbols` (asume retardo cero y fase 0); medir la EVM sobre los bordes (descartar ≥ span símbolos).
- **Verificar:** `pytest tests/test_stimuli.py -v`; abrir una figura guardada.
- **Listo cuando:** los tests pasan y hay figuras en `docs/golden_model/figuras/`.

---

### Parte A · ISI genérica

#### A1 · FIR y bloque `ISI` con taps arbitrarios — M

- **Objetivo:** `ISI.process` y `ISI.process_hw` para `(n,)` y `(2, n)`, con retardo declarado.
- **Depende de:** P1.
- **Para qué:** el FIR es la estructura del hardware. Todos los perfiles solo cambian los h[k] que se cargan (Ref. A.2), así que la estructura se valida primero con taps arbitrarios.
- **Archivos:** + `emulador/isi/{__init__,fir}.py`, ~ `emulador/blocks/isi.py`, + `tests/test_isi.py`.
- **Qué hacer:**
  1. **Modelo** (Ref. A.1, PDF Ec. 8): y[n] = Σ<sub>k=0</sub><sup>K−1</sup> h[k]·x[n−k], con taps a f<sub>s</sub>.
  2. **`fir.py`:**
     - `isi_apply(x, h)`: FIR causal del mismo largo que x, sobre el último eje, con `scipy.signal.lfilter(h, 1, x, axis=-1)`;
     - `isi_apply_stream(chunks, h)`: procesa una lista de bloques y conserva las últimas K−1 muestras de estado con `zi` (arranca en ceros y pasa `zf` al bloque siguiente) (PDF §3.6).
  3. **Bloque `ISI`** (campos en §2.3):
     - en `__post_init__`, con `profile="taps"`, `self.h = np.asarray(taps, complex)`, y validar 1 ≤ K ≤ 32;
     - `delay_samples = int(np.argmax(abs(h)))` (cursor k₀);
     - `process_hw(x) = isi_apply(x, h)`: causal, retrasado k₀, como el hardware;
     - `process(x)`: salida alineada. Rellenar x con k₀ ceros al final, filtrar y descartar las primeras k₀ muestras, así el largo se mantiene. El retardo **fraccionario** no se corrige (le toca al receptor);
     - `coefficients() = {"h": h}`.
  4. Actualizar el docstring del esqueleto: quitar el TODO de retardo y documentar la convención.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Bypass | `ISI()(x)` | igual a x (exacto) |
  | Impulso | `process_hw(δ[n])` | primeros K valores = h (exacto) |
  | Contra `numpy.convolve` | h complejo aleatorio con K = 7, 16 y 32 | `convolve(x, h)[:n]` (10⁻¹²) |
  | Por bloques | x partido en trozos de largos distintos | igual al procesamiento completo (10⁻¹²) |
  | Linealidad | ISI(a·x₁ + b·x₂) contra a·ISI(x₁) + b·ISI(x₂), con a y b complejos | 10⁻¹² |
  | Retardo | h = [0,1; 0,2; 1; 0,3] | `delay_samples == 2`; `process` = `process_hw` adelantado 2 (exacto) |
  | Dos polarizaciones | x de forma (2, n) | cada fila igual al caso (n,) (exacto) |
  | Límite de K | taps de largo 33 | `ValueError` |

- **Notebook:** — (el 02 arranca en A2).
- **Errores a evitar:** `np.convolve(..., "same")`, que centra el filtro y rompe la causalidad; recortar k₀ en `process_hw`; olvidar `axis=-1`.
- **Verificar:** `pytest tests/test_isi.py tests/test_contract.py -v`; `python -m scripts.run_block isi --param "taps=[1, 0.3j]"`.
- **Listo cuando:** los tests pasan y `run_block` guarda la figura.

#### A2 · Perfiles eco, gaussiano y medido — M

- **Objetivo:** el método de diseño en frecuencia y los perfiles de fase lineal.
- **Depende de:** A1, P2.
- **Para qué:** el gaussiano tiene fórmula cerrada y es simétrico, así que es el perfil de **validación** del método que después usan Bessel, Butterworth y la respuesta medida (Ref. A.2–A.3, PDF §3.5.3).
- **Archivos:** + `emulador/isi/profiles.py`, ~ `emulador/blocks/isi.py`, ~ `tests/test_isi.py`.
- **Qué hacer:**
  1. `normalize_energy(h)`: h/√Σ\|h\|² (PDF Ec. 9).
  2. `echo(echoes, K=None)`:
     - h[0] = 1 y h[dᵢ] = aᵢ;
     - K por defecto = max(dᵢ) + 1;
     - validar \|aᵢ\| < 1 y dᵢ ≤ 31;
     - normalizar.
  3. `gaussian(B, fs, K, L=8192)`:
     - H<sub>G</sub>(f) = exp(−(ln 2/2)·f²/B²) en `fftfreq(L, 1/fs)`;
     - h̃ = `fftshift(ifft(H)).real`;
     - truncar centrado: h[k] = h̃[L/2 − ⌊K/2⌋ + k] (PDF Ec. 10–11);
     - normalizar.
  4. `measured(H, fs, K, L=8192)` con el mismo método. H puede ser una función `H(f)` o una tabla `(f, H)`: se interpola magnitud en dB y fase desenrollada con `np.interp`, y vale 0 fuera del rango.
  5. `K_rule(B, Rs, m)`: 32 si B ≤ 0,4·R<sub>s</sub> y m ≥ 4; 16 si no.
  6. Auxiliar `energy_captured(profile, K)`: diseñar con K = L (sin truncar) y calcular la fracción de energía en los K coeficientes elegidos.
  7. En el bloque:
     - `profile="echo" | "gaussian" | "measured"`;
     - B = `B_rel`·R<sub>s</sub> y f<sub>s</sub> = R<sub>s</sub>·sps;
     - K = `K` o `K_rule(...)`.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Energía | todos los perfiles | Σ\|h\|² = 1 (10⁻¹²) |
  | −3 dB en B | 20·log10(\|H(B)\|/\|H(0)\|) con H(f) = Σ h[k]e^(−j2πfkT<sub>s</sub>), para B/R<sub>s</sub> = 0,3 … 0,8 | −3,0 ± 0,3 dB |
  | Simetría | h[k₀−i] y h[k₀+i] | iguales (10⁻¹²), k₀ = ⌊K/2⌋ |
  | Truncamiento | `energy_captured`, mismos casos | ≥ 98,5 % |
  | Escala con el escenario | mismo B/R<sub>s</sub> a 32 y 100 GBd | mismo h (10⁻¹²) |
  | Medido = gaussiano | `measured` con H<sub>G</sub> | igual a `gaussian` (10⁻¹²) |
  | Eco | `echo([(2, 0.5)])` | antes de normalizar, h = [1, 0, 0,5] |

- **Notebook:** `02_isi_generica.ipynb` (primera parte): modelo, normalización, magnitud y coeficientes del gaussiano. Incluye por qué no se muestrea h(t): figura del −3 dB corrido por aliasing (PDF, recuadro de §3.5.3).
- **Errores a evitar:** medir el −3 dB en valor absoluto (la normalización por energía cambia \|H(0)\|, siempre relativo a f = 0); muestrear la h(t) continua; olvidar `fftshift` antes de truncar el gaussiano.
- **Verificar:** `pytest tests/test_isi.py -k "gaussian or echo or measured" -v`.
- **Listo cuando:** los tests pasan y `ISI(profile="gaussian")` corre en `run_block`.

#### A3 · Perfiles Bessel y Butterworth — M

- **Objetivo:** perfiles causales desde el prototipo analógico, incluido el **perfil principal** (Bessel m = 4, B = 0,6 R<sub>s</sub>, K = 16).
- **Depende de:** A2.
- **Para qué:** representan componentes reales. El Bessel tiene fase casi lineal más un retardo; el Butterworth agrega ISI por fase y es el perfil de estrés. Diseñarlos en frecuencia evita la deformación de la transformación bilineal cerca de Nyquist (Ref. A.3, PDF §3.5.3, Tablas 2–3).
- **Archivos:** ~ `emulador/isi/profiles.py`, ~ `emulador/blocks/isi.py`, ~ `tests/test_isi.py`, + `tests/data/isi_bessel_m4_B06_K16.npy`.
- **Qué hacer:**
  1. `analog(kind, B, fs, m, K, d=None, L=4096)`:
     - `kind` es `"bessel"` o `"butter"`;
     - prototipo: `signal.bessel(m, 2πB, analog=True, norm="mag")` o `signal.butter(m, 2πB, analog=True)`.
  2. Evaluar H<sub>P</sub> en la grilla: `signal.freqs(b, a, worN=2π·fftfreq(L, 1/fs))`.
  3. Pre-retardo: multiplicar por e^(−j2πf·d·T<sub>s</sub>), con d = 3 si K = 16 y d = 6 si K = 32 (PDF Ec. 12). Conserva la oscilación previa al pulso que genera el recorte en Nyquist.
  4. `ifft`, parte real, primeros K valores y normalizar.
  5. En el bloque: `profile="bessel" | "butter"`, con `order` = m y K por `K_rule`.
  6. Generar y revisar a ojo (gráfico) los coeficientes del perfil principal; recién entonces guardarlos como `.npy` de regresión.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | −3 dB en B | m = 2 … 6; K por regla; a 2 sps B/R<sub>s</sub> = 0,3 … 0,5 y el perfil principal; a 4 sps (K y d al doble) B/R<sub>s</sub> = 0,3 … 0,8 (decisión 19) | −3,0 ± 0,3 dB |
  | Truncamiento | `energy_captured`, mismos casos | ≥ 98,5 % |
  | Fidelidad en magnitud | \|H<sub>FIR</sub>(f)/H<sub>FIR</sub>(0)\| contra \|H<sub>P</sub>(f)\| en \|f\| ≤ B, m ≥ 3, B ≤ 0,7·R<sub>s</sub> (decisión 19) | < 0,5 dB |
  | Fidelidad en retardo | τ<sub>FIR</sub>(f) − d contra τ<sub>P</sub>(f), en muestras, misma banda | < 0,4 muestras |
  | Regla de K | (0,4 R<sub>s</sub>, m=4) → 32; (0,6 R<sub>s</sub>, m=4) → 16; (0,3 R<sub>s</sub>, m=3) → 16 | exacto |
  | Regresión del perfil principal | contra el `.npy` | 10⁻¹² |
  | `norm="mag"` | Bessel m=4 | \|H<sub>P</sub>(B)\| = −3 dB (10⁻⁶) |

- **Notebook:** `02_isi_generica.ipynb`:
  - efecto del orden m en magnitud, retardo de grupo y respuesta al escalón (Tabla 2 y Fig. 1 del PDF);
  - comparación entre diseño en frecuencia y bilineal (sobrepico 0,8 % contra 15 %; 0,04 dB contra 0,53 dB), que documenta la decisión 6.
- **Errores a evitar:**
  - `signal.bessel(m, Wn)` sin `analog=True` (bilineal);
  - Bessel sin `norm="mag"` (B deja de ser el −3 dB);
  - comparar retardos sin descontar d;
  - guardar la regresión antes de que pasen los demás tests (congelaría un error).
- **Verificar:** `pytest tests/test_isi.py -k "bessel or butter or K_rule" -v`; `python -m scripts.run_block isi --param profile=bessel`.
- **Listo cuando:** los tests pasan y está la figura de diseño en frecuencia contra bilineal.

#### A4 · Perfil CD corta — S

- **Objetivo:** cargar en el FIR los taps de Savory para pocos km, que vinculan ISI y CD.
- **Depende de:** A1 y B1 (`savory_taps` y `cd_R` viven en `emulador/cd/physics.py`).
- **Para qué:** permite comparar el FIR truncado con el método en frecuencia de la CD, y es el "FIR truncado" de la Fig. 9 del PDF.
- **Archivos:** ~ `emulador/isi/profiles.py`, ~ `emulador/blocks/isi.py`, ~ `tests/test_isi.py`.
- **Qué hacer:**
  1. En B1 ya existe `savory_taps(fs, D, lam, z)`: h[n] = √(jcT<sub>s</sub>²/(Dλ²z))·exp(−jπcT<sub>s</sub>²n²/(Dλ²z)), con n = −(R−1)/2 … (R−1)/2 (PDF Ec. 24–26). Es coherente con el signo positivo: es la inversa de exp(+jπDλ²z f²/c).
  2. `cd_short(fs, D, lam, z)`: llama a `savory_taps`, exige R ≤ 32 y normaliza. A 64 GSa/s, R ≤ 31 equivale a z ≲ 55 km con SSMF.
  3. En el bloque, `profile="cd_short"` con `D_ps_nm_km`, `wavelength_nm` y `length_km` como parámetros extra. Si el bloque crece mucho, pasarle un dict `profile_params`.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Largo | 10 km a 64 GSa/s | K = R = 5 (exacto) |
  | Magnitud constante | antes de normalizar | \|h[n]\| iguales (10⁻¹²) |
  | Fase cuadrática | ajuste de grado 2 a `unwrap(angle(h))` | residuo 10⁻¹⁰ |
  | Signo | fase de H<sub>FIR</sub> a baja frecuencia contra `cd_transfer` | mismo signo de curvatura |
  | Límite | z = 100 km a 64 GSa/s (R = 55) | `ValueError` |
  | Contra la referencia | NRMSE contra `cd_reference` | **a definir**: medir en A4 y agregarlo a §1 |

- **Notebook:** `02_isi_generica.ipynb`: coeficientes del chirp y comparación con la CD en frecuencia.
- **Errores a evitar:** usar el signo de Savory tal cual sin verificarlo contra `cd_transfer` (Savory escribe el compensador).
- **Verificar:** `pytest tests/test_isi.py -k cd_short -v`.
- **Listo cuando:** los tests pasan y el criterio contra la referencia quedó agregado a §1.

#### A5 · Caracterización de la ISI — L

- **Objetivo:** reproducir la Tabla 4 del PDF y cerrar el notebook 02.
- **Depende de:** A3, P4.
- **Para qué:** confirma de punta a punta (estímulo → ISI → receptor → métrica) que los perfiles producen la ISI esperada. Además deja cuantificado el efecto del retardo fraccionario, que las tesis de temporización tienen que corregir.
- **Archivos:** + `notebooks/02_isi_generica.ipynb` (completo), ~ `tests/test_isi.py`, opcional + `scripts/sweep_isi.py`.
- **Qué hacer:**
  1. **SNR equivalente determinista:** p[n] = `symbol_rate_response(h, …)` con roll-off 0,1; SNR<sub>eq</sub> = −10·log10(`isi_power_ratio(p)`).
     - "Instante óptimo": diseñar el perfil con `Scenario(sps=16)` y el mismo B/R<sub>s</sub>, y elegir la fase de muestreo que maximiza SNR<sub>eq</sub>.
     - "Grilla fija": a 2 sps, con la fase 0 después de descontar el retardo entero.
  2. Barrido B/R<sub>s</sub> ∈ {0,4; 0,5; 0,6} × {gaussiano, Bessel m=4, Butterworth m=4}, con tabla contra el PDF.
  3. Constelación QAM16 y ojo PAM4 para cada B con Bessel m=4, usando `comm_signal` → `ISI` → `recover`.
  4. Reportar D<sub>p</sub>, la potencia de ISI relativa al cursor y \|H<sub>ISI</sub>(f)\| en banda (PDF §6.3, "magnitudes derivadas").
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Tabla 4 del PDF | SNR<sub>eq</sub> en el instante óptimo para los 9 casos | ±0,5 dB (**tolerancia propuesta, a confirmar**) |
  | Orden entre perfiles | B = 0,4 R<sub>s</sub> | Butterworth < Bessel ≈ gaussiano |

- **Notebook:** `02_isi_generica.ipynb` con la estructura de la Ref. §7 (objetivo, teoría, parámetros, simulación, validación, gráficos, conclusiones) y las figuras de la Ref. A.7.
- **Errores a evitar:** medir en la grilla fija y compararlo con la Tabla 4 (que es en el instante óptimo); usar roll-off 0,2 (el de `gen_signal`) en vez de 0,1.
- **Verificar:** `jupyter nbconvert --execute --inplace notebooks/02_isi_generica.ipynb` sin errores; `pytest`.
- **Listo cuando:** el notebook corre limpio, las figuras están guardadas y el cambio de estado del bloque 01 en el README está propuesto por PR.

---

### Parte B · Dispersión cromática

#### B1 · Física de la CD — M

- **Objetivo:** β₂, β₃, H<sub>CD</sub> con signo positivo, R, N y taps de Savory, en SI y como funciones puras.
- **Depende de:** P1, P2.
- **Para qué:** es el corazón matemático del bloque. El test de signo y el de pendiente del retardo de grupo **fijan** las decisiones 2 y 3: si alguien cambia el signo o reintroduce el ½, fallan (Ref. B.1–B.2, §3).
- **Archivos:** + `emulador/cd/{__init__,physics}.py`, + `tests/test_chromatic_dispersion.py`.
- **Qué hacer:**
  1. `betas(D, lam, S=0.0)`, con k = λ²/(2πc) (PDF Ec. 16):
     - β₂ = −D·k;
     - β₃ = k²·(S + 2D/λ);
     - si S = 0, devolver β₃ = 0 exacto (decisión 4: con S = 0 el modelo es el de solo β₂).
  2. `cd_transfer(L, fs, D, lam, z, S=0.0)` = exp(−j(β₂ω²/2 + β₃ω³/6)z), con ω = 2π·`fftfreq(L, 1/fs)` (PDF Ec. 18). Con β₃ = 0 da exp(+jπDλ²z f²/c).
  3. `cd_R(fs, D, lam, z)` = 2·⌊\|D\|λ²z/(2cT<sub>s</sub>²)⌋ + 1 (PDF Ec. 26).
  4. `cd_N(fs, D, lam, z, N_min=128, N_max=HardwareParams().N_max)` = max(N_min, 2^⌈log₂R⌉), con `ValueError` si supera N<sub>max</sub> (decisión 5).
  5. `savory_taps(fs, D, lam, z)` (ver A4), sin límite de largo.
  6. `cd_summary(esc, D, z, S)`: dict con β₂ [ps²/km], β₃ [ps³/km], pendiente del retardo de grupo [ps/GHz], R, N y L<sub>D</sub> para T₀ = 1/R<sub>s</sub> (PDF Tabla 15).
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | **Signo positivo** | fase de H contra +πDλ²z f²/c para \|f\| < 2 GHz, z = 10 km (fase < π) | 10⁻¹² |
  | Pasa-todo | \|H\| para z = 100 y 2000 km | 1 (10⁻¹²) |
  | Pendiente de τ(f) | `polyfit(f, tau, 1)` en \|f\| < f<sub>s</sub>/4 contra −Dλ²z/c | rel. < 0,1 % (detecta ½ y signo) |
  | Composición | H(z₁)·H(z₂) contra H(z₁+z₂) | 10⁻¹² |
  | Parábola con β₃ | S = 0,057; ajuste de grado 2 de τ(ω) contra β₂z y β₃z/2 | rel. < 1 % |
  | Compatibilidad | S = 0 contra modelo sin β₃ | idéntico (exacto) |
  | Valores de tabla | β₂ = −21,7 ps²/km; β₃ ≈ 0,13 ps³/km | rel. < 1 % |
  | R | 40 GSa/s: 10 km → 3, 500 km → 109; 64 GSa/s: 10/100/500/1000/2000 km → 5/55/279/559/1117 | exacto |
  | N | R ≤ 128 → 128; 500 km → 512; 2000 km → 2048; N<sub>max</sub> = 1024 con 2000 km | exacto / `ValueError` |

- **Notebook:** `03_cd_modelo_fisico.ipynb` (primera parte): magnitud (= 1), fase parabólica y retardo de grupo lineal para 10/50/100 km (Fig. 4 del PDF) y tabla de R. Completa `00_…` con las magnitudes derivadas.
- **Errores a evitar:**
  - D en ps/nm/km dentro de `physics.py` (todo en SI);
  - `np.sign` o el ½ de la guía original;
  - ajustar la pendiente en toda la banda (los bordes tienen errores numéricos del gradiente);
  - usar `3e8` (cambia algún R de tabla).
- **Verificar:** `pytest tests/test_chromatic_dispersion.py -k "sign or group_delay or cd_R or cd_N" -v`.
- **Listo cuando:** los tests pasan y `cd_summary` reproduce la Tabla 15 del PDF.

#### B2 · Referencia ideal con FFT completa — M

- **Objetivo:** la "verdad física" contra la que se mide el error del overlap-save.
- **Depende de:** B1.
- **Para qué:** separa el error del **método** (bloques finitos) del error de **cuantización** que aparece después (Ref. §1 y B.3, PDF §4.7.3).
- **Archivos:** + `emulador/cd/reference.py`, ~ `tests/test_chromatic_dispersion.py`.
- **Qué hacer:**
  1. `cd_reference(x, fs, D, lam, z, S=0.0, full=False)`:
     - pad = R + 64 ceros a cada lado;
     - L = 2^⌈log₂(len)⌉;
     - `ifft(fft(xp, L) · cd_transfer(L, …))`.
  2. Con `full=False`, recortar al largo de x y alinear (y[n] corresponde a x[n]); con `full=True`, devolver las L muestras del buffer, incluidas las colas (decisión 18).
  3. Operar sobre el último eje para aceptar `(2, n)`.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Identidad | z = 0 | NRMSE ~10⁻¹⁵ |
  | Energía | `full=True`, z = 500 km | Σ\|y\|² = Σ\|x\|² (rel. 10⁻¹²) |
  | Reversibilidad | (D, S) y luego (−D, −S) con `full=True`, 4096 símbolos | NRMSE < 10⁻⁵ (decisión 18) |
  | Pulso gaussiano | T₀ = 15,6 ps, f<sub>s</sub> ≥ 256 GSa/s, z = 25…100 km; `rms_width(z)/rms_width(0)` contra √(1 + (z/L<sub>D</sub>)²) | rel. < 1 % |

- **Notebook:** `03_cd_modelo_fisico.ipynb`:
  - ensanchamiento del gaussiano (Fig. 5 del PDF);
  - constelaciones QAM16 con CD de 1 y 5 km, y 500 km + H* (Fig. 6);
  - chirp h<sub>T</sub>[n] para 100 km (Fig. 7).
- **Errores a evitar:** hacer `ifft(fft(x)·H)` sin relleno (las colas dan la vuelta); muestrear el gaussiano a 64 GSa/s con T₀ ≈ T<sub>s</sub> (aliasing: usar f<sub>s</sub> alta o T₀ ≥ 3·T<sub>s</sub>); medir la energía sobre la salida recortada.
- **Verificar:** `pytest tests/test_chromatic_dispersion.py -k "reference or gauss" -v`.
- **Listo cuando:** los tests pasan y las Figs. 5–7 están reproducidas.

#### B3 · Overlap-save y bloque `ChromaticDispersion` — L

- **Objetivo:** el algoritmo que ejecutará la FPGA, integrado al `REGISTRY`.
- **Depende de:** B2.
- **Para qué:** es el modelo por bloques en flotante. Su error contra la referencia es el error del método; su salida cruda (`process_hw`) es la que se exporta para el RTL (Ref. B.4, PDF §4.7.2, Fig. 8).
- **Archivos:** + `emulador/cd/overlap_save.py`, ~ `emulador/blocks/chromatic_dispersion.py`, ~ `tests/test_chromatic_dispersion.py`.
- **Qué hacer:**
  1. **`cd_hbar(N, fs, D, lam, z, S=0.0)`** (lo calcula el host):
     - L = 2N;
     - H[k] = `cd_transfer(L, …)`;
     - H̄[k] = H[k]·e^(−j2πk(N/2)/L), con k = `fftfreq(L)·L` (PDF Ec. 28).
  2. **`cd_overlap_save(x, Hbar)`** (lo hace la FPGA). Para cada bloque b:
     - ventana de L = [N anteriores | N nuevas], con el primer bloque precedido por ceros;
     - FFT, producto por H̄ e IFFT;
     - descartar las primeras N muestras y entregar las N últimas.
     - La salida queda retrasada N/2.
  3. **Bloque**, con campos según §2.3:
     - `__post_init__`: convertir con `D_si`/`S_si`, λ·10⁻⁹ y z·10³; N = `N` o `cd_N(...)`; validar N par ≥ 2; precalcular `self.Hbar`;
     - `delay_samples = N // 2`; `coefficients() = {"Hbar": Hbar, "N": N}`;
     - `process_hw(x)`: `cd_overlap_save` recortado a len(x), fila por fila si la forma es (2, n);
     - `process(x)`: agregar N/2 ceros al final, aplicar OLS y tomar [N/2 : N/2 + n];
     - `from_scenario(esc, **kw)`.
  4. Corregir el docstring del esqueleto: signo **positivo**, fórmula con β₃, `fs=64e9` por defecto y referencia a §1.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Contra la referencia | ruido complejo de 20 000 muestras; z = 100/500/1000/2000 km; N = `cd_N`; comparar y[N/2 : N/2+m] con r[:m], m = n − 2N | NRMSE < 10⁻² |
  | Retardo declarado | `delay_samples` | N/2 (exacto) |
  | Alineada contra cruda | `process` contra `process_hw` desplazado N/2 | exacto |
  | Identidad | `length_km=0` | ~10⁻¹⁵; `test_contract` pasa |
  | Pasa-todo de H̄ | \|H̄\| | 1 (10⁻¹²) |
  | Dos polarizaciones | (2, n) | cada fila igual al caso (n,) |
  | Compatibilidad β₃ | S = 0 | H̄ idéntico al del modelo sin β₃ |

- **Notebook:** `04_cd_overlap_save.ipynb` (primera parte): diagrama del algoritmo, el retardo N/2 visto en la salida y la elección automática de N.
- **Errores a evitar:**
  - quedarse con las primeras N muestras de la IFFT en vez de las últimas;
  - olvidar el factor de retardo en H̄ (el chirp no causal se "parte" en la ventana);
  - usar N < R;
  - comparar contra la referencia sin descontar N/2.
- **Verificar:** `pytest tests/test_chromatic_dispersion.py tests/test_contract.py -v`; `python -m scripts.run_block chromatic_dispersion --param length_km=100`.
- **Listo cuando:** los tests pasan y `run_block` muestra la constelación dispersada.

#### B4 · Alcance del método y residuo de β₃ — L

- **Objetivo:** reproducir las Figs. 9 y 10 del PDF y cerrar los notebooks 03 y 04.
- **Depende de:** B3, P4. A4 es opcional (curva del FIR truncado).
- **Para qué:**
  - la Fig. 9 justifica la regla N ≥ R y dimensiona N<sub>max</sub> para el hardware;
  - la Fig. 10 justifica la decisión 4 (incluir β₃): muestra la ISI residual que queda cuando el compensador solo invierte β₂ (PDF §6.2).
- **Archivos:** ~ `notebooks/03_cd_modelo_fisico.ipynb`, + `notebooks/04_cd_overlap_save.ipynb`, ~ `tests/test_chromatic_dispersion.py`, opcional + `scripts/sweep_cd.py`.
- **Qué hacer:**
  1. **Barrido de alcance:** NRMSE de OLS contra la referencia con QAM16 a 32 GBd y 2 sps, z = 0 … 2000 km, N ∈ {128, …, 2048}, y la curva del FIR de Savory. Esperado: N=128 hasta ≈ 200 km, 256 hasta 300–400, 512 hasta 800–900, 1024 hasta ≈ 1500.
  2. **Residuo de β₃:**
     - canal `cd_reference(D, S=0,057)` seguido de compensación `cd_reference(−D, S=0)`;
     - `recover` y `evm_opt`;
     - R<sub>s</sub> ∈ {32, 64, 100, 200} GBd a 1000 km (y 500/2000 km para la curva);
     - usar la referencia y no OLS, porque a 400 GSa/s R ≈ 22 000.
  3. **Sección de hardware (Ref. B.8):** tasa R<sub>CD</sub> = P·f<sub>clk</sub>/2 y tabla de N necesario contra z y R<sub>s</sub>, con los parámetros provisorios marcados como tales.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Residuo de β₃ | 1000 km, 32/64/100 GBd | EVM ≈ 0,7 % / 5,6 % / 22 % (±5 % relativo) |
  | Residuo a 200 GBd | 1000 km | EVM > 100 % |
  | Control sin β₃ | S = 0 en canal y compensación | EVM en el piso del RRC (≈ 0,1 %) |

- **Notebook:** `04_cd_overlap_save.ipynb` con la Fig. 9 (NRMSE contra z) y la Fig. 10 (constelaciones y SNR equivalente contra R<sub>s</sub>).
- **Errores a evitar:** barridos largos dentro de los tests (el test cubre solo los puntos de tabla; las curvas van al notebook o script); comparar EVM sin corregir la fase.
- **Verificar:** el notebook corre limpio con `nbconvert --execute`; `pytest`.
- **Listo cuando:** figuras guardadas, conclusiones escritas y cambio de estado del bloque 03 en el README propuesto por PR.

---

### Cuantización y exportación de vectores

#### C1 · Cerrar parámetros de hardware — S

- **Objetivo:** confirmar el modelo de RFSoC, P, f<sub>clk</sub> y N<sub>max</sub>.
- **Depende de:** información externa (dirección de tesis, equipo de RTL).
- **Para qué:** el modelo bit a bit de la CD tiene que replicar las etapas radix-2² y el escalado de la FFT elegida (Ref. B.8). Sin esto, C3 no puede cerrarse para la CD.
- **Archivos:** ~ `emulador/config.py` (`HardwareParams`), ~ fila 7 de §1.
- **Qué hacer:**
  1. Relevar: dispositivo (DS889), DSP disponibles y f<sub>clk</sub> alcanzable.
  2. Fijar P según la tasa de emulación deseada: R<sub>CD</sub> = P·f<sub>clk</sub>/2.
  3. Fijar N<sub>max</sub> según la distancia máxima emulada y la tabla de B4.
  4. Actualizar la fila 7 de §1 y `HardwareParams` con `provisional=False`.
- **Tests:** `HardwareParams()` por defecto = valores decididos (exacto).
- **Notebook:** `04_cd_overlap_save.ipynb`, sección de hardware.
- **Errores a evitar:** decidir P solo por recursos sin fijar primero la tasa objetivo.
- **Verificar:** `pytest tests/test_config.py`.
- **Listo cuando:** la fila 7 de §1 queda "aceptada" y los valores provisorios quedaron reemplazados.

#### C2 · Utilidades de cuantización — S

- **Objetivo:** `quantize(x, m, n, rounding="half_up", saturate=True)` para complejos.
- **Depende de:** P2. Antes hay que decidir la convención Qm.n: si m incluye o no el bit de signo; Jiang usa Q2.8 = 10 bits, con signo incluido.
- **Para qué:** es la herramienta de la exploración de bits (Ref. C.3). Es independiente de dónde termine viviendo el modelo bit a bit.
- **Archivos:** + `emulador/quantization.py`, + `tests/test_quantization.py`.
- **Qué hacer:**
  1. Paso q = 2⁻ⁿ; rango [−2^(m−1), 2^(m−1) − q] con m bits enteros incluido el signo (a confirmar).
  2. Redondeo `half_up` = `floor(x/q + 0.5)·q`, como el hardware típico; `"trunc"` = `floor(x/q)·q`.
  3. Saturar al rango; aplicar por separado a la parte real y a la imaginaria.
  4. `to_int(x, m, n)` para obtener los enteros en complemento a dos (se usan en C4 para los `.hex`).
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Representables | valores múltiplos de q dentro del rango | sin cambio (exacto) |
  | Error de redondeo | valores aleatorios dentro del rango | \|e\| ≤ q/2 |
  | Saturación | ±10 con Q2.8 | extremos del rango (exacto) |
  | I/Q independientes | x = a + jb | igual a cuantizar a y b por separado |
  | `half_up` | 0,5·q y −0,5·q | q y 0 (no half-even) |

- **Notebook:** `05_cuantizacion.ipynb` (primera parte).
- **Errores a evitar:** `np.round`, que redondea "half-even" y no coincide con el hardware.
- **Verificar:** `pytest tests/test_quantization.py -v`.
- **Listo cuando:** los tests pasan y la convención Qm.n está agregada a §1.

#### C3 · Exploración de bits (ISI y CD) — M

- **Objetivo:** curvas NRMSE contra bits, separando referencia → flotante → cuantizado.
- **Depende de:** C2, A3 y B3. La réplica de la FFT de la CD depende además de C1 y de la decisión 14 de §1 (ubicación del modelo bit a bit).
- **Para qué:** dimensiona los anchos de palabra del RTL con un criterio numérico (Ref. C.3).
- **Archivos:** + `notebooks/05_cuantizacion.ipynb`. El modelo bit a bit va donde se decida: golden o `models/fixed_point/emulador-real/`.
- **Qué hacer:**
  1. **ISI:**
     - cuantizar la entrada (Qa.b) y h[k] (Qc.d);
     - productos a precisión completa y acumulador con ⌈log₂K⌉ bits de crecimiento;
     - redondear la salida;
     - barrer b y d, graficar NRMSE contra la salida flotante.
  2. **CD, primera aproximación** (sin réplica de FFT): cuantizar la entrada y H̄[k], y hacer la FFT en flotante. Punto de partida: Q2.8 y H̄ de 10 bits, NRMSE ≈ 2·10⁻³ hasta 400 km (Jiang).
  3. **CD, réplica radix-2² por etapas con escalado:** recién después de C1.
  4. Agregar los formatos elegidos a §1.
- **Tests:** a definir junto con los formatos (por ejemplo, NRMSE < umbral para el formato elegido).
- **Notebook:** `05_cuantizacion.ipynb`: curvas NRMSE contra bits para h[k] y H̄[k], y crecimiento del acumulador.
- **Errores a evitar:** comparar el cuantizado contra la referencia ideal en vez de contra el flotante (mezcla error de método y de cuantización).
- **Verificar:** el notebook corre limpio.
- **Listo cuando:** los formatos Qm.n están agregados a §1.

#### C4 · Exportación de vectores — M

- **Objetivo:** casos reproducibles para la verificación del RTL.
- **Depende de:** A3, B3 y P3 (PRBS). Los `.hex` dependen además de C2 y C3.
- **Para qué:** `verification/` compara el RTL contra estos archivos. Cada caso tiene que poder regenerarse bit a bit desde su `meta.json` (Ref. C.4).
- **Archivos:** + `emulador/export.py`, + `scripts/export_vectors.py`, + `tests/test_export.py`, + `notebooks/06_vectores_rtl.ipynb`.
- **Qué hacer:**
  1. `export_case(block, signal, out_dir)` escribe:
     - `entrada.npy`;
     - `coeficientes.npy` (h o H̄);
     - `salida_esperada.npy`, que es la de **`process_hw`** porque el RTL tiene retardo;
     - `meta.json`.
  2. `meta.json` incluye:
     - `scenario.to_dict()`;
     - parámetros del bloque (`dataclasses.asdict`);
     - semilla y orden de PRBS;
     - `delay_samples`;
     - versión del paquete y `git rev-parse HEAD`;
     - fecha;
     - formato Qm.n (null mientras no exista).
  3. `scripts/export_vectors.py --block isi --case bessel_m4_B06 …` con casos predefinidos (bypass, eco, Bessel principal, CD 100/500/1000 km).
  4. Destino: `outputs/vectores/<isi|cd>/<caso>/`, que no se versiona. Se regenera con el script.
  5. Cuando haya formatos Qm.n, sumar `.hex` para `$readmemh`: una línea por muestra, I y Q en complemento a dos. El formato de línea se acuerda con verificación.
- **Tests:**

  | Test | Cómo | Criterio |
  |---|---|---|
  | Ida y vuelta | exportar y cargar con `np.load` | idéntico (exacto) |
  | Metadatos | claves obligatorias de `meta.json` | todas presentes |
  | Reproducible | exportar dos veces con la misma semilla | archivos idénticos (bytes) |
  | Coherencia | `process_hw(entrada)` | = `salida_esperada` (exacto) |

- **Notebook:** `06_vectores_rtl.ipynb`: casos, formato de archivos y cómo los consume `verification/`.
- **Errores a evitar:** exportar la salida alineada (`process`) en vez de la del hardware; versionar los vectores en git.
- **Verificar:** `pytest tests/test_export.py -v`; `python -m scripts.export_vectors --block cd --case z100`.
- **Listo cuando:** existen los casos predefinidos y el formato de `meta.json` está acordado con quien haga `verification/`.

---

## 4. Checklist

Cada casilla es una tarea atómica: un commit y un push (ver §3.0). Se marca en el mismo commit que la completa.

**Base común**
- [x] P0.a · Entorno y línea de base en ASUS
- [x] P0.b · Entorno y línea de base en LENOVO
- [x] P0.c · Extra `notebooks` en `pyproject.toml` + `nbstripout --install` (ASUS)
- [x] P0.d · `pip install -e ".[dev,notebooks]"` + `nbstripout --install` en LENOVO (no requiere commit)
- [x] P1.1 · `config.py` (`C`, `D_si`, `S_si`, `Scenario`, `HardwareParams`) + `tests/test_config.py`
- [x] P1.2 · Notebook `00_escenario_y_convenciones` (primera parte)
- [x] P2.1 · `metrics.py`: `nrmse`, `group_delay` + tests
- [x] P2.2 · `metrics.py`: `evm_opt`, `snr_eq_db`, `rms_width` + tests
- [x] P2.3 · `metrics.py`: `peak_distortion`, `isi_power_ratio`, `symbol_rate_response` + tests
- [x] P3.1 · `stimuli/prbs.py` (PRBS-15/31) + tests
- [x] P3.2 · `stimuli/mapper.py` (Gray PAM4/QAM16) + tests
- [x] P3.3 · `stimuli/pulse.py`, `CommSignal`, `comm_signal` + tests
- [x] P3.4 · `stimuli/test_signals.py` (impulso, pulso gaussiano, tono) + tests
- [x] P4.1 · `rx.py` (`matched_filter`, `recover`) + tests (back-to-back, retardo, fase)
- [x] P4.2 · `plots.py` + `style.mplstyle`
- [x] P4.3 · Notebook `01_estimulos`

**Partes A y B, intercaladas** (decisión 17 de §1: A = ISI, B = CD)

*Hito 1 · Bloque ISI con taps y física de la CD*
- [x] A1.1 · `isi/fir.py` (`isi_apply`, `isi_apply_stream`) + tests
- [x] A1.2 · Bloque `ISI` con taps (`delay_samples`, `process`, `process_hw`, `coefficients`) + tests + `run_block`
- [x] B1.1 · `cd/physics.py`: `betas`, `cd_transfer` + tests de signo, pasa-todo y composición
- [x] B1.2 · Tests de pendiente del retardo de grupo y parábola con β₃
- [x] B1.3 · `cd_R`, `cd_N` + tests de tabla
- [x] B1.4 · `savory_taps`, `cd_summary` + tests

*Hito 2 · Perfiles de ISI y referencia ideal de la CD*
- [x] A2.1 · `isi/profiles.py`: `normalize_energy`, `echo`, `K_rule` + tests
- [x] A2.2 · `isi/profiles.py`: `gaussian`, `energy_captured` + tests
- [x] A2.3 · `isi/profiles.py`: `measured`; perfiles echo/gaussian/measured en el bloque + tests
- [x] B2.1 · `cd/reference.py` + tests de identidad, energía y reversibilidad
- [x] B2.2 · Test del pulso gaussiano
- [x] B2.3 · Notebook `03_cd_modelo_fisico`

*Hito 3 · Perfiles analógicos, CD corta y bloque `ChromaticDispersion`*
- [x] A3.1 · `isi/profiles.py`: `analog` (Bessel/Butterworth) + tests de −3 dB y energía
- [x] A3.2 · Tests de fidelidad contra el prototipo analógico
- [x] A3.3 · Perfiles bessel/butter en el bloque + regresión del perfil principal (`tests/data/`)
- [ ] A4.1 · Perfil `cd_short` (requiere B1.4) + tests
- [ ] B3.1 · `cd/overlap_save.py` (`cd_hbar`, `cd_overlap_save`) + test contra la referencia
- [ ] B3.2 · Bloque `ChromaticDispersion` (docstring corregido, `fs=64e9`) + tests + `run_block`

*Hito 4 · Caracterización de la ISI y alcance del overlap-save*
- [ ] A5.1 · Test de SNR equivalente contra la Tabla 4 del PDF
- [ ] A5.2 · Notebook `02_isi_generica`
- [ ] B4.1 · Tests del residuo de β₃ (Tabla 17 del PDF)
- [ ] B4.2 · Notebook `04_cd_overlap_save` (Figs. 9 y 10, hardware)

**Cuantización y exportación**
- [ ] C1.1 · Confirmar RFSoC, P, f<sub>clk</sub> y N<sub>max</sub>; actualizar `HardwareParams` y la fila 7 de §1
- [ ] C2.1 · Definir la convención Qm.n (fila nueva en §1)
- [ ] C2.2 · `quantization.py` + tests
- [ ] C3.1 · Exploración de bits de la ISI (notebook `05_cuantizacion`)
- [ ] C3.2 · Exploración de bits de la CD
- [ ] C4.1 · `export.py` + tests
- [ ] C4.2 · `scripts/export_vectors.py` con casos predefinidos
- [ ] C4.3 · Notebook `06_vectores_rtl`

**Coordinación con el equipo**
- [ ] Comunicar la convención de signo positiva a la tesis de CDC
- [ ] PR al equipo: extensión de `Block` (decisión 9)
- [ ] PR al equipo: extra `notebooks` (decisión 11), `package-data` de `style.mplstyle` y módulo `stimuli/` (decisión 10)
- [ ] PR al equipo: estado de los bloques 01 y 03 en el README

---

## 5. Puntos de integración con el equipo

Todo lo que sigue toca código o configuración compartidos: requiere acuerdo o PR hacia `main`.

| Punto | Qué se toca | Cómo se maneja |
|---|---|---|
| Interfaz `Block` | `emulador/common.py`: agregar `delay_samples`, `coefficients()` y `process_hw()` con valores por defecto neutros | Decisión 9 de §1. Hasta aceptarse, se define solo en ISI y CD |
| Contrato por defecto | `tests/test_contract.py` | No se modifica. ISI y CD lo cumplen con bypass y z = 0 |
| Utilidades nuevas | `emulador/config.py` (`Scenario`), `metrics.py`, `stimuli/`, `rx.py`, `plots.py` | Archivos nuevos que otros bloques pueden usar (AWGN, ruido de fase y PMD también necesitan f<sub>s</sub>). Avisar al crearlos |
| Señal de prueba | `common.gen_signal` (β = 0,2, no Gray, sin PRBS) contra `stimuli/` (β = 0,1, Gray, PRBS) | No tocar `gen_signal`; proponer a futuro una sola fuente de estímulos |
| Dependencias | `pyproject.toml`: extra `notebooks` (`jupyterlab`, `ipykernel`, `jupytext`, `nbstripout`, `nbconvert`) | Decisión 11 de §1; PR |
| Documentación | `README.md` del paquete: tabla de estado, "Convenciones abiertas" (signo de la CD ya decidido) | PR al cerrar A5 y B4 |
| Definición de SNR | Bloque AWGN (otra persona) | La normalización Σ\|h\|² = 1 de la ISI supone potencia conservada; coordinar |
| Tesis de CDC | Signo de H<sub>CD</sub> | Comunicar: su filtro usa la conjugada |
| Punto fijo | `models/fixed_point/emulador-real/` | Decidir con el equipo si el modelo bit a bit vive ahí (§6) |
| Verificación | `verification/` consume `outputs/vectores/` | Acordar el formato de `meta.json` y `.hex` |

---

## 6. Riesgos y decisiones abiertas

| Tema | Riesgo o pregunta | Propuesta | Estado |
|---|---|---|---|
| Modelo de RFSoC, P, f<sub>clk</sub>, N<sub>max</sub> | Bloquea C3 (réplica radix-2²) y la exportación bit a bit | Dejar N<sub>max</sub> = 2048 provisorio | abierta |
| N<sub>max</sub> en escenarios rápidos | A 200 GSa/s y 1000 km, R ≈ 5450 > 2048 | `cd_N` falla explícitamente; N<sub>max</sub> viene de `HardwareParams` | abierta |
| Ubicación del modelo bit a bit | La Ref. lo pone en el golden; el repo tiene `models/fixed_point/emulador-real/` | Utilidades y exploración en el golden, modelo bit a bit en `fixed_point` | **requiere decisión** (afecta al equipo) |
| Formatos Qm.n | Sin definir | Arrancar de Q2.8 / 10 bits (Jiang) | abierta |
| Tolerancia de la Tabla 4 (SNR eq. de la ISI) | El PDF no da tolerancia | ±0,5 dB | abierta |
| EVM back-to-back | Con roll-off 0,1 y span 32 el piso es 0,58 % (medido en P3.3) | span 64: piso 0,12 % (decisión 15) | cerrada |
| Criterio de "CD corta" contra la referencia | El PDF no lo define | Definir al hacer A4 | abierta |
| Retardo fraccionario de Bessel/Butterworth | Afecta la EVM medida si el receptor no busca el instante óptimo | `rx.py` con búsqueda del instante (sps = 16 para análisis fino) | mitigado en P4 |
| Versionado de notebooks | Diffs ilegibles y conflictos entre ASUS y LENOVO | `nbstripout` instalado en cada máquina + pares `jupytext` (percent) | abierta |
| Duplicación de estímulos | `gen_signal` y `stimuli/` divergen | Unificar por PR cuando el equipo defina la interfaz de señal | abierta |
| Interfaz común provisoria | El equipo puede cambiar `Block` | Mantener la lógica en funciones puras (`isi/`, `cd/`) y los bloques como envoltorios finos | mitigado |
| Rendimiento del OLS en Python | Lento para señales muy largas en barridos | Vectorizar por bloques si hace falta; no afecta la corrección | bajo |
