# DEC-20260928-2038-asus: Estímulos en un módulo nuevo emulador/stimuli/
- Estado: propuesta
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
`common.gen_signal` genera símbolos aleatorios no Gray, con roll-off 0,2 y sin PRBS. La referencia pide PRBS-15/31, mapper Gray, roll-off 0,1 y señales de prueba física (impulso, pulso gaussiano, tonos). `common.py` es compartido.

## Decisión
Crear `emulador/stimuli/` con `prbs.py`, `mapper.py`, `pulse.py` (reusa `common.rrc_taps`) y `test_signals.py`, sin modificar `common.py`. `gen_signal` se mantiene para `test_contract` y `run_block`.

Confirmación de Wanda: sí (2026-09-28); se usa en `wanda_dev`. Pendiente: acuerdo del equipo para unificar las fuentes de estímulos en `main`.

## Alternativas consideradas
- Ampliar `gen_signal` en `common.py`: toca código compartido y cambia el comportamiento por defecto para todos.

## Consecuencias (código, tests, notebooks, hardware)
- Código: paquete nuevo; riesgo de duplicación con `gen_signal`.
- Tests: `tests/test_stimuli.py`.
- Notebooks: `01_estimulos.ipynb`.
- Hardware: la PRBS es la misma que generará el RTL.

## Referencias
- referencia_guia_ISI_CD.md C.1; GUIA_ISI_CD.md paso P3.
