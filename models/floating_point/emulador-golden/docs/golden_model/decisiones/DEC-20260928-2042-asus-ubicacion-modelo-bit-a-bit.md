# DEC-20260928-2042-asus: Ubicación del modelo bit a bit (punto fijo)
- Estado: propuesta
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
La referencia ubica la cuantización y el modelo bit a bit dentro del golden (`cuantizacion.py`). El repositorio tiene una carpeta específica para punto fijo, `models/fixed_point/emulador-real/`, que todavía no arrancó.

## Decisión (propuesta)
- En el golden quedan `emulador/quantization.py` (utilidades Qm.n) y la exploración de bits (notebook 05).
- El modelo bit a bit de cada bloque (FIR con acumulador, FFT radix-2² con escalado) va en `models/fixed_point/emulador-real/`.

**No confirmada por Wanda.** No se aplica hasta que la confirme y lo acuerde con el equipo.

## Alternativas consideradas
- Todo en el golden, como propone la referencia: más simple, pero contradice la estructura del repo.
- Todo en `fixed_point`, incluida la exploración: separa la exploración del modelo flotante contra el que se compara.

## Consecuencias (código, tests, notebooks, hardware)
- Define dónde se implementan los pasos C3 y C4 de la guía.
- Equipo: la estructura de `fixed_point` la comparte todo el equipo.

## Referencias
- referencia_guia_ISI_CD.md §1 y C.3; README.md de la raíz; GUIA_ISI_CD.md §6.
