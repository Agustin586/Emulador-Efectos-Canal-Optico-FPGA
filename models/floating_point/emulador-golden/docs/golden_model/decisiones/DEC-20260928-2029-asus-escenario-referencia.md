# DEC-20260928-2029-asus: Escenario de referencia configurable (32 GBd, 2 sps, 64 GSa/s)
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
Varias magnitudes de ISI y CD dependen de la velocidad del sistema: f_s, T_s, el B de los perfiles, R y N de la CD. Si cada módulo fija su propia f_s, cambiar de escenario obliga a tocar todo el código.

## Decisión
- Escenario de referencia: R_s = 32 GBd, sps = 2, f_s = R_s·sps = 64 GSa/s, λ = 1550 nm, roll-off del RRC = 0,1.
- Es **configurable**: todo se deriva de un objeto `Scenario` (`emulador/config.py`), inmutable y con rangos validados (R_s entre 1 y 300 GBd; sps ∈ {1, 2, 4, 8, 16}; λ en banda C).
- El escenario se guarda en los metadatos de cada simulación y vector exportado.

## Alternativas consideradas
- f_s fija en cada bloque (así están hoy los esqueletos: `fs=2e9`): descartada, no escala a ≥ 100 GBd.
- Escenario a 1 sps: descartado, R se reduce pero la señal RRC no se representa bien.

## Consecuencias (código, tests, notebooks, hardware)
- Código: `emulador/config.py` nuevo; `ChromaticDispersion.fs` pasa a `64e9` por defecto; los bloques ofrecen `from_scenario`.
- Tests: `tests/test_config.py`; los perfiles de ISI se testean a 32 y 100 GBd.
- Notebooks: la celda de parámetros de cada notebook crea el `Scenario`.
- Equipo: otros bloques (AWGN, ruido de fase, PMD) también usan f_s; conviene que adopten el mismo `Scenario`.

## Referencias
- referencia_guia_ISI_CD.md §2; PDF Fundamentos, Tabla 12.
- GUIA_ISI_CD.md, paso P1.
