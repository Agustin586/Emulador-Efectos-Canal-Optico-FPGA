# DEC-20260928-2037-asus: Extender common.Block con delay_samples, coefficients() y process_hw()
- Estado: propuesta
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
DEC-20260928-2036-asus define estas tres piezas para ISI y CD. Si viven solo en esos dos bloques, el resto del emulador (scripts, exportación, verificación) no puede tratar a todos los bloques de forma uniforme. `emulador/common.py` es compartido y el README pide no editarlo sin avisar.

## Decisión
Proponer al equipo agregar a `Block`, con valores por defecto neutros:
- `delay_samples` → 0;
- `coefficients()` → `{}`;
- `process_hw(x)` → `self.process(x)`.

Así ningún bloque existente cambia de comportamiento.

Confirmación de Wanda: sí (2026-09-28). Pendiente: acuerdo del equipo mediante un PR a `main`. Hasta entonces, en `wanda_dev` las piezas se definen solo en ISI y CD, y `common.py` no se toca.

## Alternativas consideradas
- Dejarlo solo en ISI y CD: funciona, pero no es uniforme.
- Un mixin aparte: agrega complejidad sin beneficio claro.

## Consecuencias (código, tests, notebooks, hardware)
- Código: unas 10 líneas en `common.py`; `run_block` y `export` podrían usarlo para cualquier bloque.
- Tests: agregar a `test_contract.py` que `process_hw` desplazado `delay_samples` sea igual a `process`.
- Equipo: requiere PR y aprobación.

## Referencias
- GUIA_ISI_CD.md §5; DEC-20260928-2036-asus.
