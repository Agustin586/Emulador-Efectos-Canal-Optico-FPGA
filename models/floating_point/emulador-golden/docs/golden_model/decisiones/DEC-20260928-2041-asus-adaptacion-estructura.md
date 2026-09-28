# DEC-20260928-2041-asus: Adaptación de la estructura de la referencia al repositorio
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
La referencia propone una estructura genérica (`base.py`, `estimulos/`, `isi/bloque.py`, `vectores/`) con identificadores en castellano. El repositorio ya tiene su estructura y sus convenciones.

## Decisión
Prevalece el repositorio. El detalle está en GUIA_ISI_CD.md §2:
- Identificadores en inglés; docstrings y comentarios en castellano.
- Se usa `common.Block` en lugar de `Bloque`. Los bloques siguen en `emulador/blocks/` y en el `REGISTRY`.
- Las funciones puras de diseño y cálculo van en `emulador/isi/` y `emulador/cd/` (en SI).
- Los campos de los bloques van en unidades de hoja de datos con sufijo, y se convierten a SI adentro.
- Con los parámetros por defecto el efecto es nulo (contrato): 100 km y Bessel m = 4 son casos de ensayo, no valores por defecto.
- Los vectores van en `outputs/vectores/` (ya ignorado); las figuras en `docs/golden_model/figuras/`.

## Alternativas consideradas
- Aplicar la estructura de la referencia tal cual: rompe las convenciones y el contrato del equipo.

## Consecuencias (código, tests, notebooks, hardware)
- Módulos nuevos en el paquete compartido (`config.py`, `metrics.py`, `rx.py`, `plots.py`, `stimuli/`, `isi/`, `cd/`, `quantization.py`, `export.py`): avisar al equipo al crearlos.
- Tests: `test_contract.py` sigue siendo obligatorio.

## Referencias
- referencia_guia_ISI_CD.md §4; GUIA_ISI_CD.md §2.
