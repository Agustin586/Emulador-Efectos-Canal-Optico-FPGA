# Golden models del emulador de canal óptico

Modelos en **punto flotante (Python)** de cada deterioro del emulador de efectos de canal óptico en FPGA (Tesis 4.3.13, Proyecto Marco UNS / Fundación Fulgor).

**Para qué sirve.** Antes de escribir Verilog, cada bloque se modela acá con matemática "exacta". Ese modelo es la referencia contra la que después se valida el hardware. Por ahora este repositorio cubre solo eso.

## Estructura

```
emulador/
  common.py         contrato Block, señal de prueba (PAM4/QAM16) y EVM
  blocks/
    phase_offset.py  EJEMPLO COMPLETO
    isi.py awgn.py chromatic_dispersion.py pmd.py phase_noise.py timing_error.py
                     esqueletos: modelo, parámetros y TODO documentados
scripts/run_block.py  corre un bloque y guarda un gráfico de validación
tests/test_contract.py  lo que cumplen todos los bloques
docs/REFERENCIAS.md
```

## Puesta en marcha

Dentro del contenedor (o en cualquier Python 3.10 o superior):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Resultado esperado hoy: los tests en verde y los bloques sin implementar como `skipped` (no como error).

## Probar el ejemplo

```bash
python -m scripts.run_block phase_offset --param delta_theta_rad=0.3
```

Guarda en `outputs/` un gráfico con la constelación antes y después del bloque, y el espectro.

## Estado de los bloques

| # | Bloque | Nivel | Estado |
|---|---|---|---|
| 01 | ISI genérica | 1 | esqueleto |
| 02 | AWGN | 2 | esqueleto |
| 03 | Dispersión cromática | 3 | esqueleto |
| 04 | PMD | 3 | esqueleto |
| 05 | Offset de fase | 1 | **completo (ejemplo)** |
| 06 | Ruido de fase | 2 | esqueleto |
| 07 | Error de timing | 1 | esqueleto |

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

- **Interfaz de señal:** forma de los datos, muestras por símbolo y polarizaciones. Lo de arriba es una propuesta.
- **Definición de SNR:** por muestra o por símbolo (Es/N0). Afecta al bloque AWGN.
- **Fórmula de la CD:** en las referencias consultadas la fase es `D·λ²·z·ω²/(4πc)`, sin un factor 1/2 adicional. Revisar contra la fórmula de la guía antes de implementar el bloque 03.
