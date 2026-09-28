# DEC-20260928-2036-asus: Manejo del retardo en ISI y CD (salida alineada + salida de hardware)
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
- El overlap-save de la CD retrasa la salida N/2 muestras; el FIR de la ISI la retrasa k₀ muestras, más una parte fraccionaria en Bessel y Butterworth.
- El contrato común (`common.Block`, `tests/test_contract.py`) exige salida de la misma forma y efecto nulo con los parámetros por defecto, pero no define cómo se declara un retardo.
- La referencia propone `bloque.retardo`.

## Decisión
En los bloques ISI y CD:
- `process(x)` devuelve la salida **alineada**, con el retardo entero descontado y el mismo largo que la entrada. El retardo fraccionario no se corrige: le corresponde al receptor.
- `delay_samples` (propiedad) declara el retardo entero: k₀ en ISI, N/2 en CD.
- `process_hw(x)` devuelve la salida cruda, retrasada como en el hardware. Es la que se exporta como `salida_esperada`.
- `coefficients()` devuelve lo que se carga en el hardware: `{"h": ...}` o `{"Hbar": ..., "N": ...}`.

Confirmada por Wanda el 2026-09-28. Se aplica en `wanda_dev` sin modificar `common.py`; la extensión de la clase base se trata aparte (DEC-20260928-2037-asus).

## Alternativas consideradas
- `process` con la salida cruda (retrasada): rompe `test_default_parameters_give_negligible_effect` en la CD, porque a z = 0 igual hay un retardo de N/2.
- Devolver una salida más larga: rompe el contrato de forma.

## Consecuencias (código, tests, notebooks, hardware)
- Código: los dos bloques implementan las cuatro piezas.
- Tests:
  - `process` = `process_hw` desplazado `delay_samples` (exacto);
  - `test_contract` sigue pasando.
- Vectores: `meta.json` incluye `delay_samples`.
- Equipo: comunicar la convención; los demás bloques pueden adoptarla.

## Referencias
- referencia_guia_ISI_CD.md §3 (retardos), A.5, B.4; GUIA_ISI_CD.md §2.3.
