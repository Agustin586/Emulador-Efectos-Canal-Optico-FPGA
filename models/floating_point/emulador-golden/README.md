# Golden models del emulador de canal óptico

Modelos en **punto flotante (Python)** de cada deterioro del emulador de efectos de canal óptico en FPGA (Tesis 4.3.13, Proyecto Marco UNS / Fundación Fulgor).

**Para qué sirve.** Antes de escribir Verilog, cada bloque se modela acá con matemática "exacta". Ese modelo es la referencia contra la que después se valida el hardware. Por ahora este repositorio cubre solo eso.

## Estructura

```
emulador/
  common.py         contrato Block, señal de prueba (PAM4/QAM16) y EVM
  config.py         Scenario (Rs, sps, λ, roll-off -> fs) y HardwareParams; conversión de unidades
  metrics.py        NRMSE, EVM con ganancia óptima, retardo de grupo, ancho rms, SNR equivalente de la ISI
  stimuli/          PRBS (ITU-T O.150), mapeo Gray PAM4/QAM16, conformación RRC, señales de prueba
  rx.py             receptor mínimo: filtro adaptado, fase de muestreo y retardo óptimos, EVM
  plots.py          gráficos comunes (constelación, ojo, espectro) y estilo (style.mplstyle)
  isi/              ISI: FIR causal, diseño de perfiles en frecuencia, caracterización
  cd/               CD: física (β₂, β₃, H_CD, R, N), referencia ideal, overlap-save, residuo de β₃
  blocks/
    phase_offset.py  EJEMPLO COMPLETO
    isi.py           COMPLETO (golden model)
    chromatic_dispersion.py  COMPLETO (golden model)
    awgn.py pmd.py phase_noise.py timing_error.py
                     esqueletos: modelo, parámetros y TODO documentados
notebooks/          validación y figuras de ISI y CD (ver abajo)
scripts/run_block.py  corre un bloque y guarda un gráfico de validación
tests/              test_contract.py (lo que cumplen todos los bloques) y los tests de cada módulo
docs/REFERENCIAS.md
docs/golden_model/  referencia técnica de ISI y CD, PDF de fundamentos y figuras de los notebooks
```

## Puesta en marcha

Dentro del contenedor (o en cualquier Python 3.10 o superior):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # o ".[dev,notebooks]" para correr los notebooks
pytest
```

Resultado esperado hoy: los tests en verde y los bloques sin implementar como `skipped` (no como error).

Para trabajar con los notebooks, instalar el extra `notebooks` y, una vez por clon, `nbstripout --install`: borra las salidas de los notebooks al commitear, así el repositorio solo guarda el código y las figuras van a `docs/golden_model/figuras/`.

## Probar el ejemplo

```bash
python -m scripts.run_block phase_offset --param delta_theta_rad=0.3
```

Guarda en `outputs/` un gráfico con la constelación antes y después del bloque, y el espectro.

## Estado de los bloques

| # | Bloque | Nivel | Estado |
|---|---|---|---|
| 01 | ISI genérica | 1 | **completo (golden model)**: FIR de hasta 32 taps; perfiles eco, gaussiano, Bessel, Butterworth y respuesta medida |
| 02 | AWGN | 2 | esqueleto |
| 03 | Dispersión cromática | 3 | **completo (golden model)**: overlap-save (algoritmo de la FPGA), β₂ y β₃, referencia ideal |
| 04 | PMD | 3 | esqueleto |
| 05 | Offset de fase | 1 | **completo (ejemplo)** |
| 06 | Ruido de fase | 2 | esqueleto |
| 07 | Error de timing | 1 | esqueleto |

## Notebooks de ISI y CD

Cada notebook valida un bloque contra la teoría y el informe de fundamentos (`docs/golden_model/`), con una sección de validación (`assert`) y conclusiones:

| Notebook | Contenido |
|---|---|
| `00_escenario_y_convenciones` | Escenario de referencia (32 GBd, 2 sps, 64 GSa/s), unidades, convenciones y parámetros de hardware |
| `01_estimulos` | PRBS, mapeo Gray, conformación RRC y receptor mínimo (piso de EVM back-to-back) |
| `02_isi_generica` | Perfiles de ISI, diseño en frecuencia contra transformación bilineal, SNR equivalente (Tabla 4) y retardo fraccionario |
| `03_cd_modelo_fisico` | Función de transferencia de la CD, largo del chirp, pulso gaussiano y efecto sobre QAM16 |
| `04_cd_overlap_save` | Overlap-save, alcance de cada tamaño de bloque, residuo de β₃ y dimensionamiento del hardware |
| `07_cd_parametros_cdc` | CD con los parámetros del grupo del compensador (60 GBd, 72 GSa/s, 200 km, ventana de 384) |

## Ciclo de trabajo por bloque (por ahora, dos pasos)

1. **Estudio teórico.** Leer el docstring del bloque y las referencias. Entender la fórmula, los parámetros y su rango físico realista.
2. **Golden model.** Implementar `process` en punto flotante, correr `run_block` y revisar el gráfico. Escribir al menos un test propio que compruebe una propiedad del efecto (por ejemplo, "la potencia se conserva" o "la varianza crece linealmente").

Los pasos siguientes (punto fijo, RTL, verificación, síntesis, integración) se suman a medida que se avance.

Sugerencia de trabajo: una rama de git por bloque (`git checkout -b bloque/<nombre>`) y no editar `common.py` sin avisar, porque lo usan todos.

## Contrato provisorio de interfaz

Vale hasta que se defina la interfaz común con el grupo:

- Una señal es un `np.ndarray` complejo; el último eje es el tiempo.
- Forma `(n,)` = una polarización. Forma `(2, n)` = dos polarizaciones (X, Y). El PMD exige `(2, n)`.
- Un bloque devuelve la misma forma que recibe.
- Con `enabled=False` el bloque devuelve la entrada sin tocarla (bypass).
- Con los parámetros por defecto el efecto es nulo (salvo AWGN, que agrega ruido).

## Convenciones abiertas

- **Interfaz de señal:** forma de los datos, muestras por símbolo y polarizaciones.
- **Definición de SNR:** por muestra o por símbolo (Es/N0). Afecta al bloque AWGN.
- **Signo de la CD (definido en el golden model):** convención positiva, `H_CD(f) = exp(+j·π·D·λ²·z·f²/c)` (con β₃ opcional), sin el factor 1/2 de la guía original. **El compensador (CDC) tiene que usar el conjugado, `H*`.**
- **Retardo de los bloques:** ISI y CD exponen `delay_samples`, `coefficients()` y `process_hw()` (salida cruda, como el hardware); `process()` devuelve la salida alineada. Se propone sumar estas tres piezas a `Block` con valores neutros.
